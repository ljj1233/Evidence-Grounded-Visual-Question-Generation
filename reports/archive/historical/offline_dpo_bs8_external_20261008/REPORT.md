# PART B — batch8-final external development evaluation (DEV256)

One evaluation only: `bs8_final`. Same frozen DEV256, same SeeQ (`gpt-5.6-luna`,
documented variant) and GPT-4o Phase-2.1 support rubric as Task 1. Cache-seeded from
the trajectory caches; new requests: SeeQ 40, support 40 (all development evaluation;
no training labels). Paired image bootstrap n=10000, seed 42.

## Results (DEV256, n: SeeQ 253, GPT 256)

| system | GPT Supported | GPT Unsupported | Coverage | Search | Context | Spatial |
|---|---:|---:|---:|---:|---:|---:|
| Base | 0.844 | 0.059 | 0.372 | 0.355 | 0.275 | 0.056 |
| bs1 step1000 | 0.977 (+.133 [.090,.180]) | 0.012 (−.047 [−.078,−.020]) | 0.485 (+.113 [.080,.147]) | 0.466 (+.111 [.081,.143]) | 0.210 (−.065 [−.079,−.051]) | 0.064 (+.008 ns) |
| bs1 final | 0.984 (+.141 [.098,.188]) | 0.012 (−.047 [−.078,−.016]) | 0.390 (+.017 [−.013,+.049]) | 0.360 (+.005 [−.021,+.031]) | 0.221 (−.055 [−.070,−.040]) | 0.092 (+.036 [.016,.058]) |
| **bs8 final** | **0.984 (+.141 [.098,.188])** | 0.012 (−.047 [−.078,−.020]) | **0.485 (+.112 [.079,.147])** | **0.472 (+.117 [.087,.148])** | 0.211 (−.064 [−.078,−.049]) | 0.065 (+.009 ns) |

## Reading

- **batch-8 keeps the entire GPT-4o Supported gain** (+0.141, identical CI to bs1 final)
  and the entire Unsupported reduction (−0.047).
- **batch-8 recovers the Coverage and Search gains that bs1-final had eroded**:
  bs8 final Coverage +0.112 vs bs1 final +0.017 (ns); Search +0.117 vs +0.005 (ns).
  bs8 final ≈ bs1 step1000 on every dimension.
- **Context loss is not fixed by batch-8**: −0.064 (bs8) vs −0.065 (bs1 step1000),
  both CI-significant. The question-type/context degradation is present at the first
  useful checkpoint regardless of batch size.
- Spatial: bs1-final's +0.036 is not reproduced (bs8 +0.009 ns).

So the late over-update hypothesis (H1) is confirmed for Coverage/Search, while the
Context/type degradation (H2) is not a training-process artifact and must be addressed
in pair construction (Part C–E).
