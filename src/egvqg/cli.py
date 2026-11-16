"""Unified CLI: datasets, local-model capture, baseline generation and evaluation, probing analyses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .providers import RequestBudgetExceeded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egvqg", description="Evidence-grounded VQG research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    wbc=subparsers.add_parser('whitebox-capture',help='P0 probing: greedy plain-v2 generation at batch size 1 with hidden states at prompt end, first K question tokens and question end')
    wbc.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1',help='baseline run supplying materials, plan order and reference questions')
    wbc.add_argument('--local-model-path',required=True)
    wbc.add_argument('--device',default='cuda:0')
    wbc.add_argument('--limit',type=int,default=None)
    wbc.add_argument('--k',type=int,default=4,help='number of early question tokens to keep')
    wbc.add_argument('--template',default='plain-v2',help='generation prompt template (the fixed neutral prompt)')
    wbc.add_argument('--max-new-tokens',type=int,default=1024)
    wbc.add_argument('--no-check',action='store_true',help='skip the teacher-forced consistency check of the captured states')
    cc=subparsers.add_parser('candidates-capture',help='P1: stratified materials x seeded sampled candidates with captured hidden states')
    cc.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    cc.add_argument('--local-model-path',required=True)
    cc.add_argument('--device',default='cuda:0')
    cc.add_argument('--n-skills',type=int,default=0)
    cc.add_argument('--per-skill',type=int,default=0)
    cc.add_argument('--n-candidates',type=int,default=8)
    cc.add_argument('--temperature',type=float,default=0.7)
    cc.add_argument('--top-p',type=float,default=0.8)
    cc.add_argument('--top-k',type=int,default=20)
    cc.add_argument('--k',type=int,default=4)
    cc.add_argument('--max-new-tokens',type=int,default=1024)
    cc.add_argument('--template',default='plain-v2',help='generation prompt template (the fixed neutral prompt)')
    cc.add_argument('--exclude-run',default=None,help='skip materials already captured by this run (expansion)')
    cc.add_argument('--materials-file',default=None,help='JSONL following the material contract (relative to the project root); replaces the ScienceQA reference run')
    cc.add_argument('--group-field',default='group',help='field of the materials file used as the template/group')
    cc.add_argument('--select',choices=('all','sample'),default='all')
    cc.add_argument('--max-materials',type=int,default=None)
    dtc=subparsers.add_parser('detail-capture',help='teacher-forced per-head, per-step and image/text-pooled states from an existing capture run')
    dtc.add_argument('--capture-run',required=True)
    dtc.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    dtc.add_argument('--materials-file',default=None)
    dtc.add_argument('--local-model-path',required=True)
    dtc.add_argument('--device',default='cuda:0')
    dtc.add_argument('--limit',type=int,default=None)
    dtc.add_argument('--material-ids-file',default=None,help='optional one-ID-per-line filter for a priority pass')
    dtc.add_argument('--prompts-file',default=None,help='optional JSONL material_id/prompt overrides (e.g. SAQG replay)')
    csi=subparsers.add_parser('candidate-source-init',help='freeze unique captured candidates as a baseline source run for evaluation')
    csi.add_argument('--capture-run',required=True)
    csi.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    csi.add_argument('--materials-file',default=None)
    csr=subparsers.add_parser('candidate-source-run',help='replay candidate questions and let the local model answer them')
    csr.add_argument('--capture-run',required=True)
    csr.add_argument('--local-model-path',required=True)
    csr.add_argument('--device',default='cuda:0')
    cl=subparsers.add_parser('candidate-labels',help='join candidates with a cascade evaluation run; write labels and the propensity profile (G1)')
    cl.add_argument('--source-run',required=True)
    cl.add_argument('--eval-run',required=True)
    lap=subparsers.add_parser('label-audit-prepare',help='draw the stratified audit sample of redundant candidates and render review sheets')
    lap.add_argument('--eval-run',required=True)
    lap.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    lap.add_argument('--two-class',action='store_true',help='sample redundant and non-redundant labels for a precision/recall audit')
    laf=subparsers.add_parser('label-audit-finish',help='summarise audit_reviews.jsonl (AI review) with Wilson bounds')
    lr=subparsers.add_parser('label-repair',help='VQAG stage 0: two independent judges + arbitration on a stratified sample')
    lr.add_argument('--capture-run',default=None)
    lr.add_argument('--label-file',default=None)
    lr.add_argument('--candidates',default=None,help='unified candidates.jsonl (e.g. steering output) instead of capture+labels')
    lr.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    lr.add_argument('--materials-file',default=None)
    lr.add_argument('--materials',default=None)
    lr.add_argument('--judges',default='gpt-6-luna,gpt-5.6-terra')
    lr.add_argument('--arbiter',default='DeepSeek-V4.1-Flash')
    lr.add_argument('--sample-rate',type=float,default=1.0)
    lr.add_argument('--per-parent',type=int,default=3)
    lr.add_argument('--n-parents',type=int,default=None,help='limit to the first N parent materials after filters')
    lr.add_argument('--dry-run',action='store_true')
    lr.add_argument('--max-calls',type=int,default=0)
    lr.add_argument('--seed',type=int,default=20261113)
    lr.add_argument('--concurrency',type=int,default=1,help='parallel candidate labels (API calls); caches and the call cap are thread safe')
    fs=subparsers.add_parser('factsplit',help='VQAG 1a/P2: extract facts, split A/B, build four-condition materials')
    fs.add_argument('--reference-run',default='qwen3vl8b-baseline-test1257-simple-v1')
    fs.add_argument('--materials-file',default=None)
    fs.add_argument('--n-materials',type=int,default=5)
    fs.add_argument('--conditions',default='A,B;B,A;F,F;F,none;none,F')
    fs.add_argument('--image-backend',default='render',choices=('existing','render'))
    fs.add_argument('--model',default='gpt-6-luna')
    fs.add_argument('--dry-run',action='store_true')
    fs.add_argument('--max-calls',type=int,default=0)
    mm=subparsers.add_parser('import-mmqa-materials',help='build image+paragraph materials from the downloaded MultiModalQA (generalisation check)')
    mm.add_argument('--name',required=True)
    mm.add_argument('--count',type=int,required=True)
    mm.add_argument('--seed',type=int,default=20261113)
    mm.add_argument('--project-root',default='.')
    wcm=subparsers.add_parser('whitebox-compare',help='compare two white-box capture runs of the same materials (e.g. software versions)')
    wcm.add_argument('--run-a',required=True)
    wcm.add_argument('--run-b',required=True)
    wcm.add_argument('--project-root',default='.')
    wcm.add_argument('--output-root',default='outputs')
    led=subparsers.add_parser('api-ledger',help='calls and tokens spent so far by the probing route (evaluation runs matching the given name patterns)')
    led.add_argument('--patterns',default='cand-,variants,mmqa,steer,q35')
    led.add_argument('--project-root',default='.')
    led.add_argument('--output-root',default='outputs')
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
    syn=subparsers.add_parser('synthetic-build',help='build programmatic derived/bridging/stitching materials with split single-hop questions')
    syn.add_argument('--output-dir',default='data/processed/synthetic-composition-v1')
    syn.add_argument('--per-category',type=int,default=60)
    syn.add_argument('--seed',type=int,default=20261115)
    syn.add_argument('--derived-full',choices=('sum','count'),default='sum')
    syn.add_argument('--categories',default='derived,bridging,stitching',help='comma-separated subset of derived,bridging,stitching,interval')
    syn.add_argument('--value-min',type=int,default=2)
    syn.add_argument('--value-max',type=int,default=21)
    syn.add_argument('--project-root',default='.')
    ca=subparsers.add_parser('composition-answer',help='answer full and single-hop synthetic questions with a local model and score programmatically')
    ca.add_argument('--materials',default='data/processed/synthetic-composition-v1/materials.jsonl')
    ca.add_argument('--local-model-path',required=True)
    ca.add_argument('--device',default='cuda:0')
    ca.add_argument('--prompt',choices=('direct','two-step'),default='direct')
    ca.add_argument('--max-batch',type=int,default=8)
    ca.add_argument('--max-new-tokens',type=int,default=256)
    ca.add_argument('--limit',type=int,default=None)
    cg=subparsers.add_parser('composition-generate',help='generate questions on synthetic materials and classify dual-ness with grammar labels')
    cg.add_argument('--materials',default='data/processed/synthetic-composition-v1/materials.jsonl')
    cg.add_argument('--local-model-path',required=True)
    cg.add_argument('--device',default='cuda:0')
    cg.add_argument('--n-questions',type=int,default=4)
    cg.add_argument('--temperature',type=float,default=0.7)
    cg.add_argument('--top-p',type=float,default=0.8)
    cg.add_argument('--top-k',type=int,default=20)
    cg.add_argument('--max-batch',type=int,default=8)
    cg.add_argument('--max-new-tokens',type=int,default=96)
    cg.add_argument('--limit',type=int,default=None)
    cg.add_argument('--template',default='plain-v2')
    cg.add_argument('--suppress-numbers',action='store_true')
    mg=subparsers.add_parser('material-generate',help='generate N questions per material from any materials JSONL (generic composition generator)')
    mg.add_argument('--materials',required=True)
    mg.add_argument('--local-model-path',required=True)
    mg.add_argument('--device',default='cuda:0')
    mg.add_argument('--n-questions',type=int,default=6)
    mg.add_argument('--temperature',type=float,default=0.7)
    mg.add_argument('--top-p',type=float,default=0.8)
    mg.add_argument('--top-k',type=int,default=20)
    mg.add_argument('--max-batch',type=int,default=8)
    mg.add_argument('--max-new-tokens',type=int,default=96)
    mg.add_argument('--limit',type=int,default=None)
    mg.add_argument('--template',default='plain-v2')
    gb=subparsers.add_parser('grid-build',help='E2: build the 2x2 modality grid (rule/value in text or image) with paired materials')
    gb.add_argument('--output-dir',default='data/processed/composition-grid-v1')
    gb.add_argument('--n-materials',type=int,default=30)
    gb.add_argument('--seed',type=int,default=20261115)
    gb.add_argument('--value-min',type=int,default=2)
    gb.add_argument('--value-max',type=int,default=21)
    gb.add_argument('--cells',default='TT,TI,IT,II')
    gb.add_argument('--project-root',default='.')
    ga=subparsers.add_parser('grid-answer',help='E2: answer full and single-hop questions in every modality cell (local model, scored)')
    ga.add_argument('--materials',default='data/processed/composition-grid-v1/materials.jsonl')
    ga.add_argument('--local-model-path',required=True)
    ga.add_argument('--device',default='cuda:0')
    ga.add_argument('--max-batch',type=int,default=8)
    ga.add_argument('--max-new-tokens',type=int,default=192)
    ga.add_argument('--limit',type=int,default=None)
    ga.add_argument('--prompt',default='grid-answer-v1')
    gr=subparsers.add_parser('grid-report',help='E2: per-cell accuracy, hop decomposition and paired cross-vs-same-modality contrast')
    gr.add_argument('--materials',default='data/processed/composition-grid-v1/materials.jsonl')
    gr.add_argument('--bootstrap',type=int,default=10000)
    svr=subparsers.add_parser('self-verify-run',help='E3: answer candidate questions under TI/T/I with a local model and score against computable gold')
    svr.add_argument('--pool',required=True,help='candidate JSONL with material_id, question, generated_label')
    svr.add_argument('--materials',required=True)
    svr.add_argument('--local-model-path',required=True)
    svr.add_argument('--device',default='cuda:0')
    svr.add_argument('--tiers',default='direct,two-step')
    svr.add_argument('--conditions',default='TI,T,I')
    svr.add_argument('--max-batch',type=int,default=8)
    svr.add_argument('--max-new-tokens',type=int,default=256)
    svr.add_argument('--limit',type=int,default=0)
    svr.add_argument('--early-stop',action='store_true',help='stop after a wrong full-material answer (T/I are then unnecessary)')
    svp=subparsers.add_parser('self-verify-report',help='E3: selection precision/recall per repair tier and the paired tier contrast')
    svp.add_argument('--bootstrap',type=int,default=10000)
    svp.add_argument('--pvs-selection',default=None,help='optional API PVS selection.jsonl scored on the same questions')
    svrr=subparsers.add_parser('self-verify-real-run',help='E3 real data: strong-model reference answers then local TI/T/I answering')
    svrr.add_argument('--candidates',required=True)
    svrr.add_argument('--materials',required=True)
    svrr.add_argument('--local-model-path',required=True)
    svrr.add_argument('--reference-model',default='gpt-6-luna')
    svrr.add_argument('--device',default='cuda:0')
    svrr.add_argument('--sample-per-material',type=int,default=2)
    svrr.add_argument('--max-batch',type=int,default=8)
    svrr.add_argument('--max-new-tokens',type=int,default=256)
    svrr.add_argument('--limit',type=int,default=0)
    svrp=subparsers.add_parser('self-verify-real-report',help='E3 real data: equivalence judging and selection precision/recall')
    svrp.add_argument('--equivalence-model',default='gpt-5.6-terra')
    svrp.add_argument('--pvs-selection',default=None)
    svrp.add_argument('--limit',type=int,default=0)
    psel=subparsers.add_parser('pvs-select',help='API PVS selection stage: text-sufficiency, image-sufficiency and stitching screens per candidate')
    psel.add_argument('--candidates',required=True)
    psel.add_argument('--materials',required=True)
    psel.add_argument('--workers',type=int,default=8)
    cai=subparsers.add_parser('copy-attribute',help='E4: find the attention heads that copy a context numeral into the generated question')
    cai.add_argument('--materials',default='data/processed/synthetic-composition-v1/materials.jsonl')
    cai.add_argument('--local-model-path',required=True)
    cai.add_argument('--category',default='derived')
    cai.add_argument('--n-materials',type=int,default=20)
    cai.add_argument('--max-new-tokens',type=int,default=24)
    cai.add_argument('--device',default='cuda:0')
    cab=subparsers.add_parser('copy-ablate',help='E4: silence the attributed copy heads at decode time and measure copy/dual rates')
    cab.add_argument('--heads-file',required=True)
    cab.add_argument('--materials',default='data/processed/synthetic-composition-v1/materials.jsonl')
    cab.add_argument('--local-model-path',required=True)
    cab.add_argument('--conditions',default='control,top8,top16,random8')
    cab.add_argument('--category',default='derived')
    cab.add_argument('--n-materials',type=int,default=20)
    cab.add_argument('--samples',type=int,default=4)
    cab.add_argument('--temperature',type=float,default=0.7)
    cab.add_argument('--top-p',type=float,default=0.8)
    cab.add_argument('--top-k',type=int,default=20)
    cab.add_argument('--max-new-tokens',type=int,default=40)
    cab.add_argument('--device',default='cuda:0')
    ckm=subparsers.add_parser('copy-mask',help='E4: block generated positions from attending to the context numeral tokens')
    ckm.add_argument('--materials',default='data/processed/synthetic-composition-v1/materials.jsonl')
    ckm.add_argument('--local-model-path',required=True)
    ckm.add_argument('--conditions',default='control,block_text_digits,block_text_span')
    ckm.add_argument('--category',default='derived')
    ckm.add_argument('--n-materials',type=int,default=20)
    ckm.add_argument('--samples',type=int,default=4)
    ckm.add_argument('--temperature',type=float,default=0.7)
    ckm.add_argument('--top-p',type=float,default=0.8)
    ckm.add_argument('--top-k',type=int,default=20)
    ckm.add_argument('--max-new-tokens',type=int,default=40)
    ckm.add_argument('--strength',type=float,default=-1e4)
    ckm.add_argument('--device',default='cuda:0')
    aar=subparsers.add_parser('answerability-audit',help='local answerability screen (text-only, image-only, or both) scored against consensus labels (no API)')
    aar.add_argument('--candidates',required=True)
    aar.add_argument('--materials',required=True)
    aar.add_argument('--local-model-path',required=True)
    aar.add_argument('--conditions',default='TI,T,I,0',help='0 is the blind condition answered with no material')
    aar.add_argument('--device',default='cuda:0')
    aar.add_argument('--max-batch',type=int,default=8)
    aar.add_argument('--max-new-tokens',type=int,default=256)
    aar.add_argument('--limit',type=int,default=0)
    aar.add_argument('--samples',type=int,default=1,help='local answers per condition (>=2 enables stability and agreement evidence)')
    aar.add_argument('--temperature',type=float,default=0.0,help='sampling temperature; >0 is required for stability to mean anything')
    aap=subparsers.add_parser('answerability-report',help='score the local answerability audit against the consensus labels')
    aap.add_argument('--bootstrap',type=int,default=10000)
    aap.add_argument('--pvs-selection',default='')
    cmr=subparsers.add_parser('copy-mask-report',help='E4: merge several copy-mask runs into one report')
    cmr.add_argument('--sources',required=True,help='comma-separated run ids, e.g. copy-mask-v1,copy-mask-span-v1')
    cmr.add_argument('--materials',default='')
    cmr.add_argument('--title',default='')
    cr=subparsers.add_parser('composition-report',help='composition gaps with material bootstrap and the both-hops-correct-full-wrong list')
    cr.add_argument('--bootstrap',type=int,default=10000)
    ccomp=subparsers.add_parser('composition-compare',help='compare composition gaps and dual-generation rates across model-size runs')
    ccomp.add_argument('--runs',required=True,help='label=run_id pairs, comma separated')
    inl=subparsers.add_parser('inline-report',help='inline numbers/rules in generated questions and their relation to text redundancy')
    inl.add_argument('--synthetic',nargs='*',default=[],help='label=generation_run_id pairs')
    inl.add_argument('--mmqa',nargs='*',default=[],help='label=candidates_jsonl_path pairs')
    inl.add_argument('--mmqa-materials',default='data/processed/multimodalqa/mmqa-pool-v1/pvs60-materials.jsonl')
    inl.add_argument('--min-digits',type=int,default=2,help='minimum digits for an MMQA numeric inline match')
    rsrc=subparsers.add_parser('redundancy-source',help='E1: classify why a redundant question does not need one side (leak vs anchor)')
    rsrc.add_argument('--dataset',action='append',required=True,help='label=kind:labels_path:materials_path, kind in scienceqa|mmqa|synthetic; repeatable')
    rsrc.add_argument('--judge',default='gpt-6-luna',help='vision judge model for the leak/anchor adjudication')
    rsrc.add_argument('--no-judge',action='store_true',help='programmatic screen only (no API calls)')
    rsrc.add_argument('--concurrency',type=int,default=8)
    rsrc.add_argument('--max-calls',type=int,default=0,help='hard cap on new API calls; 0 means uncapped')
    rsrc.add_argument('--min-digits',type=int,default=1,help='minimum digits for a copied text number')
    rsrc.add_argument('--manual',type=int,default=50,help='rows in the pending human-review sheet')
    rsrc.add_argument('--dry-run',action='store_true',help='count the distinct calls without sending any')
    mrc=subparsers.add_parser('mmqa-refer-compare',help='compare two MMQA generation conditions with the same label file format')
    mrc.add_argument('--control',required=True,help='control candidates.jsonl')
    mrc.add_argument('--treatment',required=True,help='treatment candidates.jsonl')
    mrc.add_argument('--materials',default='data/processed/multimodalqa/mmqa-pool-v1/pvs60-materials.jsonl')
    mrc.add_argument('--control-validity',default=None,help='optional validity_report.json for the control')
    mrc.add_argument('--treatment-validity',default=None,help='optional validity_report.json for the treatment')
    mrc.add_argument('--control-cascade',default=None,help='optional mapped official-cascade result JSON')
    mrc.add_argument('--treatment-cascade',default=None,help='optional mapped official-cascade result JSON')
    mrc.add_argument('--bootstrap',type=int,default=10000)
    lv=subparsers.add_parser('local-validity',help='run the local-validity-v1 check on a generation run')
    lv.add_argument('--source-run',required=True)
    lv.add_argument('--local-model-path',required=True)
    lv.add_argument('--device',default='cuda:0')
    lv.add_argument('--max-batch',type=int,default=8)
    lv.add_argument('--max-new-tokens',type=int,default=96)
    lv.add_argument('--limit',type=int,default=None)
    vr=subparsers.add_parser('validity-report',help='summarise local-validity rows')
    cpmp=subparsers.add_parser('composition-compare-prompts',help='compare direct and two-step composition gaps')
    cpmp.add_argument('--direct-run',required=True)
    cpmp.add_argument('--two-step-run',required=True)
    im=subparsers.add_parser('interval-mechanism',help='time-boxed behavioural + logit-lens probe of the interval-count failures')
    im.add_argument('--local-model-path',required=True)
    im.add_argument('--device',default='cuda:0')
    im.add_argument('--failures',default='outputs/mechanism-w1-confirm-v1/mechanism_samples_confirm_8b.jsonl')
    im.add_argument('--answers',default='outputs/synth-confirm-count-8b-direct-v1/answers.jsonl')
    im.add_argument('--materials',default='data/processed/synthetic-composition-confirm-count-v1/materials.jsonl')
    im.add_argument('--controls-per-expected',type=int,default=4)
    im.add_argument('--max-new-tokens',type=int,default=32)
    im.add_argument('--limit',type=int,default=None)
    im.add_argument('--skip-lens',action='store_true')
    for command in (sbinit,sbrun,sbreport,seinit,serun,sereport,nei,ner,nerep,wbc,cc,dtc,csi,csr,cl,lap,laf,lr,fs,ca,cg,mg,ga,gr,svr,svp,svrr,svrp,psel,cai,cab,ckm,cmr,aar,aap,cr,ccomp,cpmp,im,inl,rsrc,mrc,lv,vr):
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
        elif args.command in ('candidates-capture','candidate-source-init','candidate-source-run'):
            from . import candidates as C
            common={k:getattr(args,k) for k in ('run_id','project_root','output_root')}
            if args.command=='candidates-capture':
                result=C.run_candidates(**common,reference_run=args.reference_run,model_path=args.local_model_path,device=args.device,n_skills=args.n_skills,
                    per_skill=args.per_skill,n_candidates=args.n_candidates,temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,k=args.k,max_new_tokens=args.max_new_tokens,exclude_run=args.exclude_run,template=args.template,
                    materials_file=args.materials_file,group_field=args.group_field,select=args.select,max_materials=args.max_materials)
            elif args.command=='candidate-source-init':
                result=C.init_candidate_source(**common,capture_run=args.capture_run,reference_run=args.reference_run,materials_file=args.materials_file)
            else:
                result=C.run_candidate_source(**common,capture_run=args.capture_run,model_path=args.local_model_path,device=args.device)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='synthetic-build':
            from .synthetic_composition import build_synthetic_composition
            result=build_synthetic_composition(project_root=args.project_root,output_dir=args.output_dir,per_category=args.per_category,seed=args.seed,derived_full=args.derived_full,categories=tuple(x.strip() for x in args.categories.split(',') if x.strip()),value_min=args.value_min,value_max=args.value_max)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='composition-answer':
            from .synthetic_composition import run_composition_answers
            result=run_composition_answers(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                materials_path=args.materials,model_path=args.local_model_path,device=args.device,prompt=args.prompt,
                max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,limit=args.limit)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='material-generate':
            from .material_generation import run_material_generation
            result=run_material_generation(run_id=args.run_id,materials_path=args.materials,model_path=args.local_model_path,
                device=args.device,n_questions=args.n_questions,temperature=args.temperature,top_p=args.top_p,
                top_k=args.top_k,max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,limit=args.limit,
                template=args.template,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='composition-generate':
            from .synthetic_composition import run_composition_generation
            result=run_composition_generation(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                materials_path=args.materials,model_path=args.local_model_path,device=args.device,n_questions=args.n_questions,
                temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,max_batch=args.max_batch,
                max_new_tokens=args.max_new_tokens,limit=args.limit,template=args.template,suppress_numbers=args.suppress_numbers)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='grid-build':
            from .composition_grid import build_grid_materials
            result=build_grid_materials(project_root=args.project_root,output_dir=args.output_dir,
                n_materials=args.n_materials,seed=args.seed,value_min=args.value_min,value_max=args.value_max,
                cells=tuple(x.strip() for x in args.cells.split(',') if x.strip()))
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='grid-answer':
            from .composition_grid import run_grid_answers
            result=run_grid_answers(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                materials_path=args.materials,model_path=args.local_model_path,device=args.device,
                max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,limit=args.limit,prompt=args.prompt)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='grid-report':
            from .composition_grid import analyze_grid_answers
            result=analyze_grid_answers(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                materials_path=args.materials,bootstrap=args.bootstrap)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='self-verify-run':
            from .self_verification import run_self_verification
            result=run_self_verification(run_id=args.run_id,pool_path=args.pool,materials_path=args.materials,
                local_model_path=args.local_model_path,device=args.device,
                tiers=tuple(x.strip() for x in args.tiers.split(',') if x.strip()),
                conditions=tuple(x.strip() for x in args.conditions.split(',') if x.strip()),
                max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,limit=args.limit or None,
                early_stop=args.early_stop,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='self-verify-report':
            from .self_verification import analyze_self_verification
            result=analyze_self_verification(run_id=args.run_id,project_root=args.project_root,
                output_root=args.output_root,bootstrap=args.bootstrap,pvs_selection=args.pvs_selection)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='self-verify-real-run':
            from .self_verification import run_real_verification
            result=run_real_verification(run_id=args.run_id,candidates_path=args.candidates,
                materials_path=args.materials,local_model_path=args.local_model_path,
                reference_model=args.reference_model,device=args.device,
                sample_per_material=args.sample_per_material,max_batch=args.max_batch,
                max_new_tokens=args.max_new_tokens,limit=args.limit or None,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='self-verify-real-report':
            from .self_verification import report_real_verification
            result=report_real_verification(run_id=args.run_id,equivalence_model=args.equivalence_model,
                project_root=args.project_root,output_root=args.output_root,pvs_selection=args.pvs_selection,
                limit=args.limit or None)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='pvs-select':
            from .pvs import select
            result=select(run_id=args.run_id,candidates_file=args.candidates,materials_file=args.materials,
                project_root=args.project_root,output_root=args.output_root,workers=args.workers)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='copy-attribute':
            from .copy_circuit import attribute_head_copy
            result=attribute_head_copy(run_id=args.run_id,materials_path=args.materials,
                model_path=args.local_model_path,category=args.category,n_materials=args.n_materials,
                max_new_tokens=args.max_new_tokens,device=args.device,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='copy-ablate':
            from .copy_circuit import run_copy_ablation
            result=run_copy_ablation(run_id=args.run_id,heads_file=args.heads_file,materials_path=args.materials,
                model_path=args.local_model_path,
                conditions=tuple(x.strip() for x in args.conditions.split(',') if x.strip()),
                category=args.category,n_materials=args.n_materials,samples=args.samples,
                temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,device=args.device,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='copy-mask':
            from .copy_circuit import run_copy_mask
            result=run_copy_mask(run_id=args.run_id,materials_path=args.materials,
                model_path=args.local_model_path,
                conditions=tuple(x.strip() for x in args.conditions.split(',') if x.strip()),
                category=args.category,n_materials=args.n_materials,samples=args.samples,
                temperature=args.temperature,top_p=args.top_p,top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,strength=args.strength,device=args.device,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='copy-mask-report':
            from .copy_circuit import mask_report
            result=mask_report(run_id=args.run_id,
                sources=[x.strip() for x in args.sources.split(',') if x.strip()],
                materials_path=(args.materials or None),
                project_root=args.project_root,output_root=args.output_root,
                title=(args.title or None))
            print(json.dumps({'run_id':result['run_id'],
                'conditions':{k:{kk:v[kk] for kk in ('n','copy_rate','threshold_mention_rate','dual_rate','single_image_rate','faithful_threshold_rate') if kk in v} for k,v in result['conditions'].items()}},ensure_ascii=False,indent=2))
        elif args.command=='answerability-audit':
            from .self_verification import run_answerability_audit
            result=run_answerability_audit(run_id=args.run_id,candidates_path=args.candidates,
                materials_path=args.materials,local_model_path=args.local_model_path,device=args.device,
                conditions=tuple(x.strip() for x in args.conditions.split(',') if x.strip()),
                max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,
                limit=(args.limit or None),samples=max(1,args.samples),temperature=args.temperature,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='answerability-report':
            from .self_verification import report_answerability_audit
            result=report_answerability_audit(run_id=args.run_id,bootstrap=args.bootstrap,
                pvs_selection=(args.pvs_selection or None),
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps({'run_id':result['run_id'],'questions':result['questions'],
                'rules':{k:{kk:v[kk] for kk in ('selected','recall','precision','false_positive_rate')} for k,v in result['rules'].items()},
                'external':{k:{kk:v[kk] for kk in v} for k,v in (result.get('external') or {}).items()}},ensure_ascii=False,indent=2))
        elif args.command=='composition-report':
            from .synthetic_composition import analyze_composition_answers,analyze_composition_generation
            run_dir=Path(args.project_root).resolve()/args.output_root/args.run_id
            result={}
            if (run_dir/'answers.jsonl').exists():
                result=analyze_composition_answers(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,B=args.bootstrap)
            if (run_dir/'generated.jsonl').exists():
                result['generation']=analyze_composition_generation(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root)
            if not result:
                raise ValueError('run has neither answers.jsonl nor generated.jsonl')
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='mmqa-refer-compare':
            from .mmqa_refer import compare_mmqa_refer
            result=compare_mmqa_refer(run_id=args.run_id,control=args.control,treatment=args.treatment,
                materials_path=args.materials,control_validity=args.control_validity,
                treatment_validity=args.treatment_validity,control_cascade=args.control_cascade,
                treatment_cascade=args.treatment_cascade,bootstrap=args.bootstrap,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2,default=float))
        elif args.command=='local-validity':
            from .validity_check import run_local_validity
            result=run_local_validity(run_id=args.run_id,source_run=args.source_run,model_path=args.local_model_path,
                device=args.device,max_batch=args.max_batch,max_new_tokens=args.max_new_tokens,limit=args.limit,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='validity-report':
            from .validity_check import analyze_validity
            result=analyze_validity(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='inline-report':
            from .inline_analysis import analyze_mmqa_inline,analyze_synthetic_inline,combine_inline_reports
            synthetic={}
            for part in args.synthetic:
                label,source=part.split('=',1)
                synthetic[label]=analyze_synthetic_inline(run_id=source,project_root=args.project_root,output_root=args.output_root)
            mmqa={}
            for part in args.mmqa:
                label,path=part.split('=',1)
                mmqa[label]=analyze_mmqa_inline(run_id=f'{args.run_id}/mmqa-{label}',candidates_path=path,
                    materials_path=args.mmqa_materials,project_root=args.project_root,output_root=args.output_root,
                    min_digits=args.min_digits)
            result=combine_inline_reports(run_id=args.run_id,synthetic_reports=synthetic,mmqa_reports=mmqa,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='redundancy-source':
            from .redundancy_source import parse_dataset_specs, run_redundancy_source
            from .assessment_rules import load_assessment_config
            cfg=load_assessment_config(args.project_root,'configs/necessity_assessment_v2.json')
            result=run_redundancy_source(run_id=args.run_id,datasets=parse_dataset_specs(args.dataset),
                judge_model=args.judge,knowledge=cfg['knowledge'],use_judge=not args.no_judge,
                concurrency=args.concurrency,max_calls=args.max_calls,min_digits=args.min_digits,
                manual=args.manual,dry_run=args.dry_run,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='composition-compare-prompts':
            from .synthetic_composition import compare_composition_prompts
            result=compare_composition_prompts(run_id=args.run_id,direct_run=args.direct_run,two_step_run=args.two_step_run,
                project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='interval-mechanism':
            from .interval_mechanism import run_interval_mechanism
            result=run_interval_mechanism(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                failures_path=args.failures,answers_path=args.answers,materials_path=args.materials,
                model_path=args.local_model_path,device=args.device,controls_per_expected=args.controls_per_expected,
                max_new_tokens=args.max_new_tokens,limit=args.limit,skip_lens=args.skip_lens)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='composition-compare':
            from .synthetic_composition import compare_composition_runs
            runs={part.split('=',1)[0]:part.split('=',1)[1] for part in args.runs.split(',') if '=' in part}
            result=compare_composition_runs(run_id=args.run_id,runs=runs,project_root=args.project_root,output_root=args.output_root)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='detail-capture':
            from .detail_capture import run_detail_capture
            result=run_detail_capture(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,
                capture_run=args.capture_run,reference_run=args.reference_run,materials_file=args.materials_file,
                model_path=args.local_model_path,device=args.device,limit=args.limit,
                material_ids_file=args.material_ids_file,prompts_file=args.prompts_file)
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='candidate-labels':
            from .candidate_labels import build_label_table,profile_report
            common={k:getattr(args,k) for k in ('project_root','output_root')}
            build_label_table(capture_run=args.run_id,source_run=args.source_run,eval_run=args.eval_run,**common)
            print(json.dumps(profile_report(capture_run=args.run_id,eval_run=args.eval_run,**common),ensure_ascii=False,indent=2))
        elif args.command in ('label-audit-prepare','label-audit-finish'):
            from . import label_audit as LA
            common={k:getattr(args,k) for k in ('project_root','output_root')}
            result=(LA.prepare_two_class_audit(capture_run=args.run_id,eval_run=args.eval_run,reference_run=args.reference_run,**common)
                    if args.command=='label-audit-prepare' and args.two_class else
                    LA.prepare_audit(capture_run=args.run_id,eval_run=args.eval_run,reference_run=args.reference_run,**common)
                    if args.command=='label-audit-prepare' else LA.finish_audit(capture_run=args.run_id,**common))
            print(json.dumps(result,ensure_ascii=False,indent=2))
        elif args.command=='label-repair':
            from .label_repair import run_label_repair
            result=run_label_repair(capture_run=args.capture_run,label_file=args.label_file,candidates=args.candidates,
                project_root=args.project_root,output_root=args.output_root,run_id=args.run_id,
                reference_run=args.reference_run,materials_file=args.materials_file,materials=args.materials,
                judges=tuple(x.strip() for x in args.judges.split(',') if x.strip()),arbiter=args.arbiter,
                sample_rate=args.sample_rate,per_parent=args.per_parent,dry_run=args.dry_run,
                max_calls=args.max_calls,seed=args.seed,n_parents=args.n_parents,concurrency=args.concurrency)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='factsplit':
            from .factsplit import run_factsplit
            result=run_factsplit(project_root=args.project_root,output_root=args.output_root,run_id=args.run_id,
                reference_run=args.reference_run,materials_file=args.materials_file,n_materials=args.n_materials,
                conditions=args.conditions,image_backend=args.image_backend,model=args.model,
                dry_run=args.dry_run,max_calls=args.max_calls)
            print(json.dumps(result,ensure_ascii=False,indent=2,default=float))
        elif args.command=='import-mmqa-materials':
            from .dataset import import_mmqa_materials
            print(json.dumps(import_mmqa_materials(args.project_root,name=args.name,count=args.count,seed=args.seed),ensure_ascii=False,indent=2))
        elif args.command=='whitebox-compare':
            from .capture import compare_captures
            print(json.dumps(compare_captures(run_a=args.run_a,run_b=args.run_b,project_root=args.project_root,output_root=args.output_root),ensure_ascii=False,indent=2))
        elif args.command=='api-ledger':
            from .candidates import api_ledger
            print(json.dumps(api_ledger(patterns=args.patterns.split(','),project_root=args.project_root,output_root=args.output_root),ensure_ascii=False,indent=2))
        elif args.command=='whitebox-capture':
            from .capture import run_capture
            result=run_capture(run_id=args.run_id,project_root=args.project_root,output_root=args.output_root,reference_run=args.reference_run,
                model_path=args.local_model_path,device=args.device,limit=args.limit,k=args.k,max_new_tokens=args.max_new_tokens,check=not args.no_check,template=args.template)
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
        else:
            parser.error(f"unknown command: {args.command}")
            return 2
    except (RequestBudgetExceeded, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
