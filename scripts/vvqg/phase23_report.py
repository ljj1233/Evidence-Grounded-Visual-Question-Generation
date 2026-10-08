#!/usr/bin/env python3
"""Protocol-only orchestration of canonical baseline metrics on the frozen DEV set."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import compare_reference_metrics, load_references, evaluate_system
from vvqg.evaluation.common.io import read_jsonl


def traditional(pred, reference):
    return compare_reference_metrics(pred, pred, reference)['left']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', type=Path, default=ROOT/'outputs/evaluation/sft_greedy/val2014_sample1000.jsonl')
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/phase23/evaluation/baseline_metrics.json')
    args = parser.parse_args()
    reference = load_references(args.references)
    systems, greedy = {}, {}
    for name in ('zero_shot', 'sft'):
        base = ROOT/'outputs/phase23/baselines'/name
        greedy[name] = read_jsonl(base/'greedy.jsonl')
        sampled = read_jsonl(base/'sampled_4x.jsonl')
        ids = {str(r['image_id']) for r in greedy[name]}
        assert len(greedy[name]) == len(ids) == 1000 and len(sampled) == 4000
        assert {str(r['image_id']) for r in sampled} == ids
        systems[name] = {
            'greedy': {'diversity': evaluate_system(greedy[name], None, ['diversity'], 'cpu', 64)[0]['diversity']},
            'sampled_4x': {'diversity': evaluate_system(sampled, None, ['diversity'], 'cpu', 64)[0]['diversity']},
        }
    comparison = compare_reference_metrics(greedy['sft'], greedy['zero_shot'], reference)
    systems['sft']['greedy']['traditional'] = comparison['left']
    systems['zero_shot']['greedy']['traditional'] = comparison['right']
    result = {
        'schema': 'phase23-baseline-metrics-v2',
        'references': {'path': str(args.references.relative_to(ROOT)),
                       'sha256': hashlib.sha256(args.references.read_bytes()).hexdigest(),
                       'images': len(reference), 'records': sum(map(len, reference.values()))},
        'systems': systems, 'paired_sft_minus_zero_shot': comparison['delta'],
        'bootstrap_protocol': comparison['bootstrap_protocol'],
        'semantic_within_image_diversity': 'BLOCKED: official embedding checkpoint unavailable; no substitute model used.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.out), 'paired_sft_minus_zero_shot': comparison['delta']}, indent=2))


if __name__ == '__main__':
    main()
