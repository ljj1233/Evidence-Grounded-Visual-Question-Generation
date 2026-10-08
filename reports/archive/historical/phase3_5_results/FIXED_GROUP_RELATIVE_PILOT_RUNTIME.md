# Fixed-Rollout Group-Relative Policy Optimization Pilot — Runtime

Hardware: 1× NVIDIA A800-SXM4-80GB. Model: Qwen3-VL-8B-Instruct + Final SFT
LoRA. All generation uses PIL image processing, cross-image batching, bf16.

## Old/reference log-probability precompute

| Item | Value |
|---|---:|
| Responses | 8,000 (2,000 groups × 4; P1 and P2 share the cache) |
| Batching | one forward per 4-response group (matches trainer microbatch) |
| Wall time | 1,831 s (~30.5 min) |
| Throughput | ~4.4 responses/s |
| Numerical audit | max abs token-logprob error **0.0** (recompute vs cache) |
| Cache size | ~11 MB (`old_policy_cache/`) |

A first attempt at batch-32 precompute produced 0.21-nat audit error because
bf16 logits are not invariant to batch shape; per-group batching removes it.

## Training

| Run | Steps | Wall time | Peak GPU alloc | Peak GPU reserved | Trainable params |
|---|---:|---:|---:|---:|---:|
| P1 UVFC-only | 250 | 1,892 s (~31.5 min) | 27.4 GB | 32.3 GB | 43,646,976 / 8,810,770,672 (0.495%) |
| P2 UVFC+Violation | 250 | 2,234 s (~37.2 min) | 27.4 GB | 32.3 GB | same |

Per-run wall time includes step-0 parity (100 responses), five train-only
monitor generations (128 images each), checkpoint saves, and end-of-run
`development1000` generation (1 greedy + 4 sampled per image). Step-0 parity
error was 0.0 in both runs; the smoke run (32 groups, 4 steps) completed in 89 s
including parity and the directional test.

## Generation (inside the training process)

| Output | Records | Notes |
|---|---:|---|
| P1 greedy1000 | 1,000 | greedy, 1 question/image |
| P1 sampled4x1000 | 4,000 | temperature 0.8, top_p 0.9, 32 tokens |
| P2 greedy1000 | 1,000 | |
| P2 sampled4x1000 | 4,000 | |

Generation reuses the in-memory trained model, so no extra model loads.

## External evaluation

| Evaluation | P1 wall | P2 wall | API requests (initial) | Cache hits (retry) |
|---|---:|---:|---:|---:|
| SeeQ / Luna (`gpt-5.6-luna`, greedy1000) | 466 s (983/1000) | 356 s (994/1000) | 1,000 each | retry filled to 1,000/1,000 |
| GPT-4o visual support (greedy1000) | 413 s (998/1000) | 332 s (999/1000) | 1,000 each | retry filled to 1,000/1,000 |
| Traditional BLEU/CIDEr/Q-BLEU/Distinct-2/dup | ~10 s CPU | ~10 s CPU | 0 | n/a |
| Qwen3-Embedding-4B semantic diversity (8,000 questions) | 86 s | 19 s | 0 | 0 |

All failures were retried to full coverage; SeeQ and support rates use the valid
parsed denominator. No GPT/Luna judge was called during training monitoring.

## Tests

`PYTHONPATH=src .venv-vllm-cu128-v011/bin/python -m pytest -q` → **90 passed**
(103 s), including the 14 new Phase-3A checks (tie mid-rank, duplicate reward
equality, P2 lexicographic order, zero-variance skip, completion mask,
log-prob gather determinism, PPO clip sign, advantage-gradient direction, base
freeze, shared SFT init, holdout non-leakage).

## Reproduce

```bash
# prep + canonicalization + hashes
.venv/bin/python scripts/vvqg/phase3a_prepare.py
.venv/bin/python scripts/vvqg/phase3a_prepare_monitor.py
# frozen old/reference logprobs (per-group batching)
.venv/bin/python scripts/vvqg/phase3a_precompute.py
# P1 then P2 (independent processes, same SFT state)
.venv/bin/python scripts/vvqg/train_fixed_group_relative.py --config configs/phase3a/p1_uvfc_fixed_group.yaml
.venv/bin/python scripts/vvqg/train_fixed_group_relative.py --config configs/phase3a/p2_uvfc_violation_fixed_group.yaml
# comparison + decision (metrics env)
.venv-vllm-cu128-v011/bin/python scripts/vvqg/phase3a_compare.py --resamples 5000
```
