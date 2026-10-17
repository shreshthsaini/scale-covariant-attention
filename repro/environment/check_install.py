"""Record CPU import checks and package differences without loading model weights."""
from datetime import datetime, timezone
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sca.envfix import missing_empty_modules


def normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")


def main():
    installed = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}
    inherited = {}
    for line in (ROOT / "repro/generation-freeze.txt").read_text().splitlines():
        if "==" in line:
            name, version = line.split("==", 1)
            inherited[normalize(name)] = version
    missing = missing_empty_modules()
    assert not missing, f"Fresh Torch/Triton installation is incomplete: {missing}"
    modules = ["torch", "torchvision", "triton", "diffusers", "transformers", "accelerate", "peft"]
    imported = {name: str(Path(importlib.import_module(name).__file__).resolve()) for name in modules}
    from diffusers import FluxPipeline, Flux2Pipeline, FluxTransformer2DModel, Flux2Transformer2DModel
    import torch
    assert not torch.cuda.is_initialized(), "CPU checks unexpectedly initialized CUDA"
    dependency_errors = []
    for dist in metadata.distributions():
        for line in dist.requires or []:
            req = Requirement(line)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            actual = installed.get(normalize(req.name))
            if actual is None or (req.specifier and actual not in req.specifier):
                dependency_errors.append({"package": dist.metadata["Name"], "requires": line, "installed": actual})
    assert not dependency_errors, dependency_errors
    record = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "executable": sys.executable,
        "missing_zero_byte_torch_triton_modules": missing,
        "imports": imported,
        "classes": [cls.__name__ for cls in (FluxPipeline, Flux2Pipeline, FluxTransformer2DModel, Flux2Transformer2DModel)],
        "cuda_initialized": torch.cuda.is_initialized(),
        "active_nonextra_dependency_errors": dependency_errors,
        "installed": dict(sorted(installed.items())),
        "changed_from_inherited": {name: {"inherited": inherited[name], "fresh": version}
                                   for name, version in sorted(installed.items())
                                   if name in inherited and version != inherited[name]},
        "new_dependencies": {name: version for name, version in sorted(installed.items()) if name not in inherited},
        "inherited_packages_not_selected": sorted(set(inherited) - set(installed)),
        "scope": "Fresh wheel installation and CPU imports only; no learned weights loaded and no GPU generation reproduced.",
    }
    out = Path(sys.argv[1])
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ("missing_zero_byte_torch_triton_modules", "classes", "cuda_initialized", "changed_from_inherited")}, indent=2))


if __name__ == "__main__":
    main()
