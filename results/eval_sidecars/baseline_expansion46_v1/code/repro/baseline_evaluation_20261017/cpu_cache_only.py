#!/usr/bin/env python3
"""Run the unchanged distribution estimator, refusing any encoder cache miss."""
import importlib.util
from pathlib import Path


def deny_encoder(*args,**kwargs):
    raise RuntimeError('Cache-only CPU stage refuses Inception extraction; rerun GPU scoring in a new workspace')


def validate_args(args):
    metrics={x.strip() for x in args.metrics.split(',')}
    if args.device!='cpu' or not metrics or not metrics<={'fid','kid','patch_fid','patch_kid'}:
        raise ValueError('Cache-only execution requires CPU distribution metrics only')
    if args.bootstrap!=1000 or args.reference_clusters!='images':
        raise ValueError('Expected 1000 reference-image-cluster draws')


def main():
    path=Path(__file__).resolve().parents[1]/'evaluate_corrected.py'
    spec=importlib.util.spec_from_file_location('cache_only_corrected',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args=module.configure_cli();validate_args(args)
    # Guard the actual extraction constructor, including a cache miss after the barrier.
    import crope.metrics.evaluate as historical
    import crope.metrics.models as models
    module.InceptionFeatures=historical.InceptionFeatures=models.InceptionFeatures=deny_encoder
    from check_scores import main as check_scores
    workspace=Path(args.out).resolve().parent
    if Path(args.root).resolve()!=workspace/'samples' or Path(args.prompts).resolve()!=workspace/'prompts.jsonl':
        raise ValueError('CPU sample view must be the verified workspace view')
    if Path(args.references).resolve()!=workspace/'references.jsonl':
        raise ValueError('CPU references must be the verified workspace manifest')
    check_scores(workspace)
    result=module.Evaluation(args).run()
    check_scores(workspace)
    return result


if __name__=='__main__':raise SystemExit(main())
