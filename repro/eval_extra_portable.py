#!/usr/bin/env python3
"""Use archived helpers and the supplied Torch cache for extra-metric reproduction."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    spec = importlib.util.spec_from_file_location("sca_eval_extra_portable", ROOT / "scripts/eval_extra.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DEFAULT_CROPE = str(ROOT / "repro/evaluator_snapshot")
    args = module.parser().parse_args()
    module.AESTHETIC_WEIGHTS = str(Path(args.torch_home) / "hub/checkpoints/laion_aesthetic/sac+logos+ava1-l14-linearMSE.pth")
    module.main()


if __name__ == "__main__":
    main()
