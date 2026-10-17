"""Bounded, weight-free CUDA smoke check. Prepare here; schedule separately."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def compiler_preflight(cc, cxx, directory, receipt):
    """Select explicit GNU host compilers and exercise Triton's C flags on CPU."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    receipt.update(requested={"CC": cc, "CXX": cxx},
                   inherited={name: os.environ.get(name) for name in ("CC", "CXX")}, probes=[])
    selected = {}
    for variable, command in (("CC", cc), ("CXX", cxx)):
        executable = shutil.which(command)
        if executable is None:
            raise RuntimeError(f"{variable} compiler is unavailable: {command}")
        selected[variable] = str(Path(executable).resolve())
    receipt["selected"] = selected
    # Set both before importing Torch: Triton reads CC and Inductor captures CXX.
    os.environ.update(selected)
    for variable, language in (("CC", "c"), ("CXX", "cpp")):
        executable = selected[variable]
        version = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                 timeout=10, check=True)
        source = directory / f"host_probe.{language}"
        library = directory / f"host_probe_{language}.so"
        prefix = "#include <Python.h>\n" if language == "c" else "#include <vector>\n"
        prefix += "#if !defined(__GNUC__) || defined(__NVCOMPILER) || defined(__clang__)\n#error GNU compiler required\n#endif\n"
        body = "int sca_host_probe(void) { return PY_MAJOR_VERSION + 39; }\n"
        if language == "cpp":
            body = 'static_assert(__cplusplus >= 201703L);\nextern "C" int sca_host_probe(void) { return std::vector<int>{42}[0]; }\n'
        source.write_text(prefix + body)
        command = [executable, str(source), "-O3", "-shared", "-fPIC", "-Wno-psabi",
                   "-o", str(library)]
        command += ["-I" + sysconfig.get_paths()["include"]] if language == "c" else ["-std=c++17"]
        probe = {"variable": variable, "executable": executable,
                 "executable_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                 "version": version.stdout.strip(), "command": command,
                 "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        receipt["probes"].append(probe)
        print("Compiler preflight: " + json.dumps(command), flush=True)
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        probe.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if result.returncode:
            raise RuntimeError(f"{variable} compiler preflight failed: {result.stderr.strip()}")
        loaded = ctypes.CDLL(str(library.resolve()))
        loaded.sca_host_probe.restype = ctypes.c_int
        value = loaded.sca_host_probe()
        probe.update(load_result=value, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        if value != 42:
            raise RuntimeError(f"{variable} compiler probe returned {value}, expected 42")
    receipt["passed"] = True


def supervise(command, timeout, log):
    """Bound a process group created here; verify its leader before signalling."""
    with Path(log).open("w") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            # A child that has not been reaped cannot have its PID reused.
            identity = subprocess.run(
                ["ps", "-o", "pid,ppid,lstart,stat,args", "-p", str(child.pid)],
                text=True, capture_output=True, check=True, timeout=5,
            )
            stream.write("\nTimeout; verified owned process group:\n" + identity.stdout)
            stream.flush()
            if child.poll() is None and os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
            return child.returncode, True


def worker(out, cc="gcc", cxx="g++", compiler_only=False):
    start = time.monotonic()
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "stage": "imports", "checks": [],
        "scope": "Tiny random bf16 CUDA operations and model only; no learned weights, no benchmark reproduction, no cuSPARSELt-specific sparse kernel test.",
        "python": sys.executable, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(Path(__file__).with_name("generation.lock").read_bytes()).hexdigest(),
        "tolerance": {"atol": 0.02, "rtol": 0.02, "policy": "torch.allclose on finite float32-converted outputs; fixed before GPU execution"},
    }
    write_json(out, record)
    try:
        record["stage"] = "host_compiler_preflight"
        record["compiler_preflight"] = {}
        write_json(out, record)
        compiler_preflight(cc, cxx, Path(out).parent / (Path(out).stem + "_cache") / "compiler",
                           record["compiler_preflight"])
        write_json(out, record)
        if compiler_only:
            record.update(status="passed", stage="compiler_preflight_complete",
                          scope="CPU host compiler compile/load preflight only; no CUDA checks executed.",
                          elapsed_seconds=time.monotonic() - start)
            write_json(out, record)
            return 0
        import torch
        import torch.nn.functional as F
        import diffusers
        from diffusers import FluxTransformer2DModel
        sys.path.insert(0, str(ROOT))
        from sca.attn import PiTable, SCAContext, install, _FLEX
        from sca.envfix import missing_empty_modules
        assert not missing_empty_modules(), "Torch/Triton has missing zero-byte modules"
        assert torch.cuda.is_available(), "CUDA is unavailable"
        torch.set_num_threads(2)
        torch.manual_seed(20261017)
        torch.cuda.manual_seed_all(20261017)
        torch.backends.cuda.matmul.allow_tf32 = False
        props = torch.cuda.get_device_properties(0)
        record.update(torch=torch.__version__, diffusers=diffusers.__version__,
                      cuda_runtime=torch.version.cuda, gpu=props.name,
                      capability=list(torch.cuda.get_device_capability(0)),
                      gpu_memory_bytes=props.total_memory)

        def stage(name):
            record.update(stage=name, elapsed_seconds=time.monotonic() - start)
            write_json(out, record)

        def compare(name, actual, reference):
            torch.cuda.synchronize()
            a, b = actual.float(), reference.float()
            finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            passed = finite and bool(torch.allclose(a, b, atol=0.02, rtol=0.02))
            delta = a - b
            record["checks"].append({"name": name, "shape": list(a.shape), "finite": finite,
                "max_abs_diff": float(delta.abs().max()),
                "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-12)), "passed": passed})
            record["elapsed_seconds"] = time.monotonic() - start
            write_json(out, record)
            assert passed, f"{name} failed declared tolerance"

        with torch.inference_mode():
            stage("cuda_matmul")
            a = torch.randn(32, 32, dtype=torch.float32).to(torch.bfloat16) / 32 ** 0.5
            b = torch.randn(32, 32, dtype=torch.float32).to(torch.bfloat16) / 32 ** 0.5
            compare("bf16_matmul_vs_cpu_float32", a.cuda() @ b.cuda(), (a.float() @ b.float()).cuda())
            stage("cuda_sdpa")
            q, k, v = [torch.randn(1, 2, 64, 16, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
            reference = torch.softmax(q.float() @ k.float().transpose(-2, -1) / 4, -1) @ v.float()
            compare("bf16_sdpa_vs_explicit_float32", F.scaled_dot_product_attention(q, k, v), reference)
            stage("tiny_flux_model")
            model = FluxTransformer2DModel(patch_size=1, in_channels=4, num_layers=1,
                num_single_layers=1, attention_head_dim=16, num_attention_heads=2,
                joint_attention_dim=24, pooled_projection_dim=8, guidance_embeds=False,
                axes_dims_rope=(4, 6, 6)).to(device="cuda", dtype=torch.bfloat16).eval()
            ids = torch.zeros(8, 8, 3, device="cuda")
            ids[..., 1] = torch.arange(8, device="cuda")[:, None]
            ids[..., 2] = torch.arange(8, device="cuda")[None, :]
            inputs = dict(hidden_states=torch.randn(1, 64, 4, device="cuda", dtype=torch.bfloat16),
                encoder_hidden_states=torch.randn(1, 6, 24, device="cuda", dtype=torch.bfloat16),
                pooled_projections=torch.randn(1, 8, device="cuda", dtype=torch.bfloat16),
                timestep=torch.tensor([0.7], device="cuda", dtype=torch.bfloat16),
                img_ids=ids.reshape(-1, 3), txt_ids=torch.zeros(6, 3, device="cuda"), return_dict=False)

            def run(context=None):
                undo = install(model, context) if context else None
                try:
                    return model(**inputs)[0]
                finally:
                    if undo:
                        undo()

            stock = run()
            compare("tiny_flux_stock_vs_full", run(SCAContext(mode="full")), stock)
            stage("compile_native_window_flex")
            native = SCAContext(mode="sca", n0=8, tile=8, window=8, local_impl="flex",
                                fused=True, pi=PiTable("const", value=0.37))
            compare("tiny_flux_native_full_window_vs_stock", run(native), stock)
            stage("compile_sparse_flex")
            options = dict(mode="sca", n0=4, tile=4, window=4, pi=PiTable("const", value=0.6))
            gather = run(SCAContext(**options, local_impl="gather"))
            flex = run(SCAContext(**options, local_impl="flex", fused=True))
            compare("tiny_flux_sparse_fused_flex_vs_gather", flex, gather)
            record["compiled_function_cache_keys"] = sorted(_FLEX)
            assert "sca" in _FLEX and "fn" in _FLEX, "Compiled SCA/FlexAttention path was not used"
        record.update(status="passed", stage="complete", elapsed_seconds=time.monotonic() - start,
                      peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(out, record)
        return 0
    except BaseException as exc:
        record.update(status="failed", elapsed_seconds=time.monotonic() - start,
                      error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        write_json(out, record)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Durable scratch JSON output; also writes .log")
    parser.add_argument("--archive", type=Path, help="Optional WORK copy of final JSON and log")
    parser.add_argument("--timeout", type=int, default=280, help="Worker timeout, 1 to 280 seconds; cleanup allows 10 seconds")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cc", default="gcc", help="GNU C compiler for Triton host shims; inherited CC is replaced")
    parser.add_argument("--cxx", default="g++", help="GNU C++ compiler for Inductor; inherited CXX is replaced")
    parser.add_argument("--compiler-preflight-only", action="store_true", help="Compile/load C and C++ host probes on CPU, then stop before importing Torch")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 280:
        parser.error("timeout must be between 1 and 280 seconds")
    if args.worker:
        return worker(args.out, args.cc, args.cxx, args.compiler_preflight_only)
    if args.out.exists():
        parser.error("use a new output path to preserve previous evidence")
    if args.archive and (args.archive.exists() or args.archive.with_suffix(".log").exists()):
        parser.error("use a new archive path to preserve previous evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Bound compilation parallelism and keep every generated cache beside scratch evidence.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "TORCHINDUCTOR_COMPILE_THREADS"):
        os.environ[key] = "2" if key != "TORCHINDUCTOR_COMPILE_THREADS" else "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    for key, name in (("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")):
        os.environ[key] = str(args.out.parent / (args.out.stem + "_cache") / name)
    log = args.out.with_suffix(".log")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(args.out),
               "--cc", args.cc, "--cxx", args.cxx]
    if args.compiler_preflight_only:
        command.append("--compiler-preflight-only")
    rc, timed_out = supervise(command, args.timeout, log)
    result = json.loads(args.out.read_text()) if args.out.exists() else {"status": "failed", "error": "worker produced no JSON"}
    result.update(worker_exit_code=rc, timed_out=timed_out, timeout_seconds=args.timeout)
    if timed_out:
        result["status"] = "timeout"
    write_json(args.out, result)
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.out, args.archive)
        shutil.copyfile(log, args.archive.with_suffix(".log"))
    print(json.dumps({"status": result["status"], "out": str(args.out), "timed_out": timed_out}))
    return 0 if rc == 0 and not timed_out and result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
