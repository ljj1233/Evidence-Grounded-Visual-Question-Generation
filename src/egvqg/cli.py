"""Unified CLI for generation, evaluation and dataset import."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .pipeline import RunConfigMismatch, RunResult, execute_run, rescore_run, summarize_run
from .providers import RequestBudgetExceeded


def _fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "0/0"


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


def _print_result(result: RunResult, *, resumed: bool) -> None:
    metrics: Mapping[str, Any] = result.metrics
    validity = metrics["validity"]
    bdr = metrics["bdr"]
    no_material = metrics["no_material"]
    print(metrics["warning"])
    print(f"run_dir: {result.run_dir}")
    print(
        f"N={metrics['n_planned']}  "
        f"valid={_fraction(validity['n_valid'], metrics['n_planned'])} "
        f"({_percent(validity['valid_rate'])})  "
        f"invalid/no-valid={validity['n_invalid']}  "
        f"unresolved={validity['n_unresolved']}  "
        f"pending={validity['n_pending']}"
    )
    print(
        f"BDR={_fraction(bdr['n_pass'], metrics['n_planned'])} "
        f"({_percent(bdr['bdr'])}), unresolved={bdr['n_unresolved']}, "
        f"upper={_percent(bdr['bdr_upper_with_unresolved'])}"
    )
    print(
        f"no-material={_fraction(no_material['n_correct'], no_material['n_eligible'])} "
        f"({_percent(no_material['accuracy'])}), "
        f"denominator share={_percent(no_material['denominator_share_of_planned'])}, "
        f"unresolved={no_material['n_unresolved']}"
    )
    print(
        f"failure breakdown: confirmed-invalid={validity['n_invalid_confirmed']}, "
        f"generation-failure={validity['n_generation_failure']}, "
        f"technical-failure={validity['n_technical_failure']}"
    )
    if resumed:
        print("resume: existing records were reused; no duplicate questions/responses/scores were appended")
    print(f"new provider calls: {result.new_calls}; new records: {result.new_records}")
    print(f"summary: {result.run_dir / 'summary.md'}")


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/stage_a.json", help="config JSON path")
    parser.add_argument("--run-id", default=None, help="stable run identifier; same ID resumes the same config")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of materials (small-sample run)")
    parser.add_argument("--output-root", default=None, help="output root; defaults to config.run.output_root")
    parser.add_argument("--project-root", default=".", help="project root used to resolve config paths")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egvqg", description="Evidence-grounded VQG research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or resume the configured pipeline")
    _add_common_run_arguments(run_parser)

    smoke_parser = subparsers.add_parser("smoke", help="run the built-in Stage A synthetic offline smoke test")
    _add_common_run_arguments(smoke_parser)

    summarize_parser = subparsers.add_parser("summarize", help="recompute metrics from an existing run")
    summarize_parser.add_argument("--run-id", required=True, help="existing run identifier")
    summarize_parser.add_argument("--output-root", default="outputs", help="output root containing the run")
    summarize_parser.add_argument("--project-root", default=".", help="project root")

    data_parser = subparsers.add_parser("download-data", help="download/import pinned official ScienceQA candidates")
    data_parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    data_parser.add_argument("--limit", type=int, default=3, help="candidate materials to import; archive is downloaded once")
    data_parser.add_argument("--seed", type=int, default=0)
    data_parser.add_argument("--ids", nargs="+", default=None, help="explicit source IDs for diagnostic material selection")
    data_parser.add_argument("--max-download-mb", type=int, default=1024, help="maximum image archive size in MiB")
    data_parser.add_argument("--offline", action="store_true", help="import already downloaded files")
    data_parser.add_argument("--project-root", default=".")

    fetch = subparsers.add_parser("fetch-dataset", help="download and verify raw MultiModalQA/WebQA, without model runs")
    fetch.add_argument("--dataset", choices=("multimodalqa","webqa","all"), required=True)
    fetch.add_argument("--workers", type=int, default=2)
    fetch.add_argument("--proxy", default=None, help="optional HTTP(S) proxy for public WebQA/Drive downloads")
    fetch.add_argument("--image-mirror", action="store_true", help="use pinned SHA256-verified WebQA image-volume mirror")
    fetch.add_argument("--project-root", default=".")

    score_parser = subparsers.add_parser("score", help="model-score saved answers in a new resumable run")
    score_parser.add_argument("--source-run", required=True)
    score_parser.add_argument("--run-id", required=True)
    score_parser.add_argument("--limit", type=int, default=None, help="limit source materials, retaining all their methods/attempts")
    score_parser.add_argument("--max-requests", type=int, default=12, help="hard HTTP-attempt budget across resumptions")
    score_parser.add_argument("--max-tokens", type=int, default=768)
    score_parser.add_argument("--dry-run", action="store_true")
    score_parser.add_argument("--output-root", default="outputs")
    score_parser.add_argument("--project-root", default=".")




    normalize = subparsers.add_parser("normalize-materials", help="map a dataset's flat JSONL fields to the shared T/I contract")
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.add_argument("--dataset", required=True)
    normalize.add_argument("--id-field", default="id")
    normalize.add_argument("--text-field", default="text")
    normalize.add_argument("--image-field", default="image")
    normalize.add_argument("--split-field", default="split")
    normalize.add_argument("--group-field", default=None)
    normalize.add_argument("--image-root", default=None)
    normalize.add_argument("--question-field", default=None, help="optionally map an existing Q field (never an answer)")
    normalize.add_argument("--questions-output", default=None, help="paired with --question-field: output material_id/question JSONL")



    initialize = subparsers.add_parser('experiment-init', help='freeze plan and initialize one complete blank record per attempt')
    initialize.add_argument('--materials', required=True)
    initialize.add_argument('--cases', default=None, help='existing case_id/material_id/question JSONL; imports Q only')
    initialize.add_argument('--methods', nargs='+', choices=('plain','strong','anchored','no_image_anchor'), default=['plain'])
    initialize.add_argument('--partition', choices=('dev','diagnostic','pilot'), default='dev')
    initialize.add_argument('--stage', type=int, default=0)
    initialize.add_argument('--config', default='configs/experiments.json')
    initialize.add_argument('--prompt-override', action='append', default=[], metavar='ROLE=VERSION',
                            help='freeze an alternative template version in a new run; repeat per role')

    execute = subparsers.add_parser('experiment-run', help='run missing steps of a versioned planned experiment')
    execute.add_argument('--max-requests', type=int, default=180)
    execute.add_argument('--max-steps', type=int, default=None, help='pause after this many new durable steps')
    execute.add_argument('--dry-run', action='store_true')
    execute.add_argument('--methods', nargs='+', choices=('plain','strong','anchored','no_image_anchor','imported'),
                         default=None, help='activate only selected planned methods; pilot defaults to plain/strong')
    execute.add_argument('--stages', nargs='+', choices=('generation','reference','answers','scores','audit'),
                         default=None, help='run only these stages; other steps stay pending for a later resume')
    execute.add_argument('--attempt-ids', nargs='+', default=None,
                         help='run only these planned attempts; the others keep pending steps and stay in N')

    metrics = subparsers.add_parser('records-metrics', help='offline fixed-N metrics and bounds from records.jsonl')
    metrics.add_argument('--records', required=True)
    metrics.add_argument('--output-dir', default=None)
    patterns = subparsers.add_parser('records-patterns', help='print TI/T/I/0 outcome patterns per method; writes nothing')
    patterns.add_argument('--records', required=True)
    patterns.add_argument('--attempts', action='store_true', help='also list each attempt')
    refreview = subparsers.add_parser('reference-review', help='apply explicit reference decisions in a review_gated run')
    refreview.add_argument('--reviews', required=True)
    packet = subparsers.add_parser('review-packet', help='write a source-only (T/I/Q) anonymous review packet; no model calls')

    review = subparsers.add_parser('experiment-review', help='import explicit human labels; never promote automatic labels')
    review.add_argument('--reviews', required=True)
    review.add_argument('--adjudicate', action='store_true')
    freeze = subparsers.add_parser('freeze-evaluation', help='freeze after all 12 controls have human adjudication')
    freeze.add_argument('--protocol', default=None)
    probe = subparsers.add_parser('probe-reproducibility', help='three fresh dev requests, separate from cache replay')
    probe.add_argument('--max-requests', type=int, default=3)

    repeated=subparsers.add_parser('probe-repeatability',help='3 fixed dev questions x 5 fresh requests')
    repeated.add_argument('--max-requests',type=int,default=15)
    batch=subparsers.add_parser('stage-two',help='validate frozen inputs, run plain30 then strong30 and report')
    batch.add_argument('--evaluation-run',required=True)
    batch.add_argument('--materials',required=True)
    batch.add_argument('--repeatability-run',default='repeatability-3x5')
    batch.add_argument('--method',choices=('plain','strong','both'),default='both')
    batch.add_argument('--max-requests',type=int,default=1000)
    batch.add_argument('--dry-run',action='store_true')
    retry=subparsers.add_parser('retry-evaluation',help='explicitly reset failed evaluation steps without changing Q or N')
    retry.add_argument('--attempt-ids',nargs='+',required=True)
    retry.add_argument('--steps',nargs='+',required=True)
    fq=subparsers.add_parser('faithfulness-review',help='derive an offline, source-bound question faithfulness review; no model calls')
    fq.add_argument('--source-run',required=True)
    fq.add_argument('--reviews',required=True)
    bainit=subparsers.add_parser('baseline-analysis-init',help='freeze a bounded API audit of existing baseline failures and unresolved necessity')
    bainit.add_argument('--source-run',required=True)
    barun=subparsers.add_parser('baseline-analysis-run',help='execute missing fixed API audit calls; no generation')
    barun.add_argument('--dry-run',action='store_true')
    barun.add_argument('--max-steps',type=int,default=None)
    bareport=subparsers.add_parser('baseline-analysis-report',help='merge explicit targeted API proof reviews and report separate dimensions offline')
    sbinit=subparsers.add_parser('simple-baseline-init',help='freeze simple API generation+TI answering, bounded images and optional valid prior-call reuse')
    sbinit.add_argument('--materials',required=True)
    sbinit.add_argument('--model',default='gpt-6-luna')
    sbinit.add_argument('--concurrency',type=int,default=16)
    sbinit.add_argument('--max-image-edge',type=int,default=1024)
    sbinit.add_argument('--max-image-pixels',type=int,default=1000000)
    sbinit.add_argument('--max-image-bytes',type=int,default=524288)
    sbinit.add_argument('--reuse-run',default=None)
    sbrun=subparsers.add_parser('simple-baseline-run',help='async generation+TI answering only; no evaluation')
    sbrun.add_argument('--limit',type=int,default=None)
    sbrun.add_argument('--local-model-path',default=None,help='run a local Qwen3-VL model instead of the API (single GPU, dynamic batching)')
    sbrun.add_argument('--device',default='cuda:0')
    sbrun.add_argument('--max-batch',type=int,default=8)
    sbrun.add_argument('--max-new-tokens',type=int,default=1024)
    sbreport=subparsers.add_parser('simple-baseline-report',help='report generation and answering progress without quality labels')
    seinit=subparsers.add_parser('simple-baseline-eval-init',help='freeze evaluation only after all simple baseline generation+TI answers finish')
    seinit.add_argument('--source-run',required=True)
    seinit.add_argument('--model',default='gpt-6-luna')
    seinit.add_argument('--concurrency',type=int,default=16)
    seinit.add_argument('--cascade',action='store_true',help='cheap single-modality screen first; witness proposals only for items it cannot cut, no per-proof verification')
    serun=subparsers.add_parser('simple-baseline-eval-run',help='async post-batch faithfulness, original answer and necessity evaluation')
    serun.add_argument('--limit',type=int,default=None)
    sereport=subparsers.add_parser('simple-baseline-eval-report',help='report post-batch evaluation separately from generation progress')
    nei=subparsers.add_parser('necessity-ensemble-init',help='freeze a two-group + arbiter necessity evaluation from a finished simple-baseline evaluation run')
    nei.add_argument('--source-run',required=True)
    nei.add_argument('--model-a',default='gpt-6-luna')
    nei.add_argument('--model-b',default='DeepSeek-V4.1-Flash')
    nei.add_argument('--arbiter',default='gpt-5.6-terra')
    nei.add_argument('--concurrency',type=int,default=32)
    nei.add_argument('--max-http',type=int,default=16000)
    nei.add_argument('--baseline-reference',action='store_true',help="use the evaluated model's own answer as the reference (default: both groups answer independently)")
    nei.add_argument('--all-items',action='store_true',help='also evaluate items without a witness in both directions (diagnostic; they can never be confirmed dual)')
    ner=subparsers.add_parser('necessity-ensemble-run',help='async sufficiency screening, witness recompute and answer-slot extraction; arbiter only for open items')
    ner.add_argument('--phase',choices=('screen','arbitrate','all'),default='all')
    ner.add_argument('--limit',type=int,default=None)
    nerep=subparsers.add_parser('necessity-ensemble-report',help='fixed-denominator report: confirmed dual vs not confirmed, with diagnostics')
    for command in (initialize, execute, review, refreview, packet, freeze, probe, repeated, batch, retry):
        command.add_argument('--run-id', required=True)
        command.add_argument('--project-root', default='.')
        command.add_argument('--output-root', default='outputs')
    for command in (fq,bainit,barun,bareport,sbinit,sbrun,sbreport,seinit,serun,sereport,nei,ner,nerep):
        command.add_argument('--run-id',required=True);command.add_argument('--project-root',default='.')
        command.add_argument('--output-root',default='outputs')

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ('simple-baseline-eval-init','simple-baseline-eval-run','simple-baseline-eval-report'):
            from .simple_baseline_evaluation import init_evaluation,run_evaluation,report_evaluation
            common={k:getattr(args,k) for k in ('run_id','project_root','output_root')}
            if args.command=='simple-baseline-eval-init':result=init_evaluation(**common,source_run=args.source_run,model=args.model,concurrency=args.concurrency,cascade=args.cascade)
            elif args.command=='simple-baseline-eval-run':result=run_evaluation(**common,limit=args.limit)
            else:result=report_evaluation(**common)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command in ('necessity-ensemble-init','necessity-ensemble-run','necessity-ensemble-report'):
            from .necessity_ensemble import init_ensemble,run_ensemble,report_ensemble
            common={k:getattr(args,k) for k in ('run_id','project_root','output_root')}
            if args.command=='necessity-ensemble-init':
                result=init_ensemble(**common,source_run=args.source_run,model_a=args.model_a,model_b=args.model_b,arbiter=args.arbiter,
                    concurrency=args.concurrency,max_http=args.max_http,candidates_only=not args.all_items,reference='baseline' if args.baseline_reference else 'independent')
            elif args.command=='necessity-ensemble-run':result=run_ensemble(**common,phase=args.phase,limit=args.limit)
            else:result=report_ensemble(**common)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command in ('simple-baseline-init','simple-baseline-run','simple-baseline-report'):
            from .simple_baseline import init_simple_baseline,run_simple_baseline,report_simple_baseline
            common={k:getattr(args,k) for k in ('run_id','project_root','output_root')}
            if args.command=='simple-baseline-init':
                result=init_simple_baseline(**common,materials_path=args.materials,model=args.model,concurrency=args.concurrency,
                    max_image_edge=args.max_image_edge,max_image_pixels=args.max_image_pixels,max_image_bytes=args.max_image_bytes,reuse_run=args.reuse_run)
            elif args.command=='simple-baseline-run':
                factory=None
                if args.local_model_path:
                    from .local_vision import local_provider
                    local=local_provider(args.local_model_path,device=args.device,max_batch=args.max_batch,max_new_tokens=args.max_new_tokens)
                    factory=lambda item,tokens:local
                result=run_simple_baseline(**common,limit=args.limit,provider_factory=factory)
            else:result=report_simple_baseline(**common)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='baseline-analysis-report':
            from .faithfulness import report_targeted_analysis
            result=report_targeted_analysis(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command in ('baseline-analysis-init','baseline-analysis-run'):
            from .baseline_analysis import init_analysis,run_analysis
            common={k:getattr(args,k) for k in ('run_id','project_root','output_root')}
            result=(init_analysis(**common,source_run=args.source_run) if args.command=='baseline-analysis-init'
                    else run_analysis(**common,dry_run=args.dry_run,max_steps=args.max_steps))
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='faithfulness-review':
            from .faithfulness import derive_faithfulness
            result=derive_faithfulness(source_run=args.source_run,run_id=args.run_id,
                reviews_path=args.reviews,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command in ('experiment-init', 'experiment-run', 'experiment-review', 'freeze-evaluation',
                            'probe-reproducibility', 'records-metrics',
                            'probe-repeatability','stage-two','retry-evaluation',
                            'records-patterns','reference-review','review-packet'):
            from . import experiments
            common = {k: getattr(args, k) for k in ('run_id','project_root','output_root') if hasattr(args, k)}
            if args.command == 'experiment-init':
                overrides = {}
                for value in args.prompt_override:
                    role, separator, version = value.partition('=')
                    if not separator or not version or role in overrides:
                        raise ValueError('prompt overrides require unique ROLE=VERSION pairs')
                    overrides[role] = version
                result = experiments.init_experiment(**common, materials_path=args.materials, cases_path=args.cases,
                    methods=tuple(args.methods), partition=args.partition, stage=args.stage, config_path=args.config,
                    prompt_overrides=overrides)
            elif args.command == 'experiment-run':
                result = experiments.run_experiment(**common, max_requests=args.max_requests,
                                                   max_steps=args.max_steps, dry_run=args.dry_run, methods=args.methods,
                                                   stages=args.stages, attempt_ids=args.attempt_ids)
            elif args.command == 'experiment-review':
                result = experiments.apply_human_reviews(**common, reviews_path=args.reviews, adjudicate=args.adjudicate)
            elif args.command == 'reference-review':
                result = experiments.apply_reference_reviews(**common, reviews_path=args.reviews)
            elif args.command == 'review-packet':
                result = experiments.export_source_review(**common)
            elif args.command == 'records-patterns':
                from .metrics import outcome_patterns
                from .records import read_jsonl
                result = outcome_patterns(read_jsonl(Path(args.records)))
                if not args.attempts:
                    for item in result['by_method'].values():
                        item.pop('attempts')
            elif args.command == 'freeze-evaluation':
                result = experiments.freeze_evaluation(**common,protocol_path=args.protocol)
            elif args.command == 'probe-repeatability':
                result = experiments.probe_repeatability(**common,max_requests=args.max_requests)
            elif args.command == 'stage-two':
                from .batch import run_stage_two
                result=run_stage_two(**common,evaluation_run=args.evaluation_run,materials_path=args.materials,
                    repeatability_run=args.repeatability_run,method=args.method,max_requests=args.max_requests,dry_run=args.dry_run)
            elif args.command == 'retry-evaluation':
                from .batch import retry_failed_evaluation
                result=retry_failed_evaluation(**common,attempt_ids=args.attempt_ids,steps=args.steps)
            elif args.command == 'probe-reproducibility':
                result = experiments.probe_reproducibility(**common, max_requests=args.max_requests)
            else:
                from .metrics import summarize_records
                result = summarize_records(args.records, output_dir=args.output_dir)
                result = {'n_planned':result['n_planned'], 'output_dir':args.output_dir or str(Path(args.records).parent)}
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command in {"run", "smoke"}:
            result = execute_run(
                config_path=args.config,
                project_root=args.project_root,
                run_id=args.run_id,
                limit=args.limit,
                output_root=args.output_root,
            )
            _print_result(result, resumed=result.new_calls == 0 and result.new_records == 0)
        elif args.command == "download-data":
            from .dataset import import_scienceqa
            print(json.dumps(import_scienceqa(Path(args.project_root), split=args.split, limit=args.limit,
                                             seed=args.seed, max_download_mb=args.max_download_mb,
                                             offline=args.offline, source_ids=args.ids), ensure_ascii=False, indent=2))
        elif args.command == "fetch-dataset":
            from .downloads import download_multimodalqa, download_webqa
            result={}
            if args.dataset in ("multimodalqa","all"):
                result['multimodalqa']=download_multimodalqa(args.project_root,workers=args.workers)
            if args.dataset in ("webqa","all"):
                result['webqa']=download_webqa(args.project_root,workers=args.workers,proxy=args.proxy,image_mirror=args.image_mirror)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command == "normalize-materials":
            from .dataset import normalize_materials
            result = normalize_materials(input_path=Path(args.input), output_path=Path(args.output),
                                         dataset_name=args.dataset, id_field=args.id_field, text_field=args.text_field,
                                         image_field=args.image_field, split_field=args.split_field, group_field=args.group_field,
                                         image_root=Path(args.image_root) if args.image_root else None,
                                         question_field=args.question_field,
                                         questions_output=Path(args.questions_output) if args.questions_output else None)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "score":
            result = rescore_run(source_run=args.source_run, run_id=args.run_id, project_root=args.project_root,
                                 output_root=args.output_root, limit=args.limit, max_requests=args.max_requests,
                                 max_tokens=args.max_tokens, dry_run=args.dry_run)
            if isinstance(result, dict):
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                _print_result(result, resumed=result.new_calls == 0 and result.new_records == 0)
        elif args.command == "summarize":
            result = summarize_run(
                run_id=args.run_id,
                project_root=args.project_root,
                output_root=args.output_root,
            )
            _print_result(result, resumed=True)
        else:
            parser.error(f"unknown command: {args.command}")
            return 2
    except (RunConfigMismatch, RequestBudgetExceeded, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
