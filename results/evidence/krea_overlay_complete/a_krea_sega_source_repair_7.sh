#!/bin/bash
# SCA_MIN_SECONDS=3000
set -euo pipefail
# Seven source-provenance repairs; original baseline outputs are preserved.
export HF_HOME=/scratch/09032/saini_2/tmp/hf TORCH_HOME=/scratch/09032/saini_2/tmp/torch HF_HUB_OFFLINE=1 XDG_CACHE_HOME=/scratch/09032/saini_2/tmp/cache TRITON_CACHE_DIR=/scratch/09032/saini_2/tmp/triton OMP_NUM_THREADS=8
export WANDB_MODE=online WANDB_PROJECT=scale-covariant-attention WANDB_DIR=/scratch/09032/saini_2/sca/wandb
unset CROPE_ROPE
export SCA_REPAIR_TASK_PATH="$0"
export CROPE_TRANSFORMER=/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python - <<'SCA_REPAIR_PREFLIGHT'
from pathlib import Path
import hashlib, importlib.metadata, json, os
source=Path('/scratch/09032/saini_2/sca/code/crope_c05ebdc')
manifest=Path('/scratch/09032/saini_2/sca/prepared/krea_sega_source_repair_20261017/unmatched_sega_prompts.jsonl')
model=Path('/scratch/09032/saini_2/tmp/hf/hub/models--black-forest-labs--FLUX.1-Krea-dev/snapshots/8162a9c7b05a641be098422bf2fcf335615c2f28')
assert os.environ.get("CROPE_TRANSFORMER")==str(model), "Wrong Krea transformer"
assert not os.environ.get("CROPE_ROPE"), "Unexpected RoPE override"
files=["crope/"+name for name in ("__init__.py","runtime.py","gain.py","rope.py","sega_official.py","spectral.py","utils.py")]
files += ["scripts/"+name for name in ("sample.py","calibrate.py","probe_mtf.py")]
digest=hashlib.sha256()
for name in files:
    digest.update(name.encode());digest.update((source/name).read_bytes())
assert digest.hexdigest()=='fe1a23e318cf1e350d64a1236d2f7481cef5d87fd744e96b29b9be93a99a5392', "Frozen source fingerprint changed"
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='3326fa743627704c476bbbd35b4339066a15fa69b18fe3ad98bfe5ff32d0179f', "Repair manifest changed"
records=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
assert len(records)==len({row["id"] for row in records})==7
assert hashlib.sha256((model/"transformer/config.json").read_bytes()).hexdigest()=='a8d4ca92dc64ca8116c370ab91339e18a723ab950c18de513d09c3922df7a2d1', "Krea transformer config changed"
assert importlib.metadata.version("diffusers")=="0.39.0"
assert importlib.metadata.version("torch")=="2.11.0+cu128"
print("Verified seven-prompt manifest, frozen implementation, Krea transformer config, and runtime versions",flush=True)
from datetime import datetime, timezone
import platform, sys, uuid
receipt_directory=Path('/scratch/09032/saini_2/sca/out/krea_baseline_overrides/sega_official/provenance')
receipt_directory.mkdir(parents=True,exist_ok=True)
executed_task=Path(os.environ["SCA_REPAIR_TASK_PATH"]).resolve()
record={"phase":"preflight_validated_before_sampler","created_utc":datetime.now(timezone.utc).isoformat(),
        "host":platform.node(),"slurm_job_id":os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "executed_task":str(executed_task),"executed_task_sha256":hashlib.sha256(executed_task.read_bytes()).hexdigest(),
        "transformer_override":os.environ["CROPE_TRANSFORMER"],
        "transformer_config_sha256":hashlib.sha256((model/"transformer/config.json").read_bytes()).hexdigest(),
        "frozen_source_root":str(source),"implementation_sha256":digest.hexdigest(),
        "python_executable":sys.executable,"python_version":platform.python_version(),
        "environment_versions":{name:importlib.metadata.version(name) for name in ("torch","diffusers","transformers")},
        "manifest":str(manifest),"manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "prompt_ids":[row["id"] for row in records],"method":"sega_official","seed":0,"steps":28,
        "guidance":4.5,"resolution":"4096x4096","output":'/scratch/09032/saini_2/sca/out/krea_baseline_overrides/sega_official/4096x4096',
        "completion_policy":"This receipt records verified launch provenance only; completion requires seven valid image/sidecar pairs."}
receipt_path=receipt_directory/("repair_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex+".json")
temporary=receipt_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record,indent=2)+"\n")
os.replace(temporary,receipt_path)
print("Launch provenance receipt:",receipt_path,flush=True)
SCA_REPAIR_PREFLIGHT
cd /scratch/09032/saini_2/sca/code/crope_c05ebdc
/work/09032/saini_2/vista/library/uv-envs/sca/bin/python scripts/sample.py --method sega_official --res 4096x4096 --prompts /scratch/09032/saini_2/sca/prepared/krea_sega_source_repair_20261017/unmatched_sega_prompts.jsonl --start 0 --end 7 --seed 0 --steps 28 --guidance 4.5 --out /scratch/09032/saini_2/sca/out/krea_baseline_overrides/sega_official/4096x4096

# All four Krea methods must be complete in the repaired overlay.
sca_generation_rc=$?
if [ "$sca_generation_rc" -eq 0 ]; then
  if ! /work/09032/saini_2/vista/library/uv-envs/sca/bin/python /scratch/09032/saini_2/sca/dispatch/20261017_eval_promotion/promote_ready_eval.py e9_krea --scratch /scratch/09032/saini_2/sca --baselines /scratch/09032/saini_2/sca/out/krea_baseline_overlay_20261017 --apply; then
    printf '%s\n' 'WARNING: evaluation promotion failed; generation status preserved' >&2
  fi
fi
exit "$sca_generation_rc"
