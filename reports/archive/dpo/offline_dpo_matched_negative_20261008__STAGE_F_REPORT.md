# STAGE F — external evaluation of matched-negative models (DEV256)

Two systems only (CONTROL-M final, TREATMENT-M final); Base reused from cache; frozen
SeeQ (`gpt-5.6-luna`) + GPT-4o Phase-2.1 rubric. New requests: SeeQ 183, support 183
(all development evaluation). Paired image bootstrap n=10000, seed 42.

## vs Base

| metric | CONTROL-M | TREATMENT-M |
|---|---|---|
| GPT Supported | 0.930 (+0.086 [0.039, 0.133]) | 0.930 (+0.086 [0.043, 0.129]) |
| GPT Unsupported | 0.012 (−0.047 [−0.078,−0.016]) | 0.031 (−0.027 [−0.059,+0.004]) |
| Coverage | 0.381 (+0.009 [−0.018,+.034]) | 0.380 (+0.008 [−0.015,+.030]) |
| Search | 0.353 (−0.002 ns) | 0.364 (+0.010 ns) |
| Context | 0.278 (+0.003 [−0.014,+.020]) | 0.249 (−0.026 [−0.042,−0.011]) |
| Spatial | 0.089 (+0.033 [0.013,0.054]) | 0.068 (+0.012 ns) |

## TREATMENT-M vs CONTROL-M

| metric | delta | 95% CI |
|---|---|---|
| GPT Supported | +0.000 | [−0.035,+0.035] |
| GPT Unsupported | +0.020 | [+0.004,+0.039] |
| Coverage | −0.001 | [−0.025,+0.023] |
| Search | +0.012 | [−0.008,+0.032] |
| Context | −0.029 | [−0.040,−0.019] |
| Spatial | −0.021 | [−0.039,−0.005] |

## Type-stratified GPT support (rate, n)

| system | yes/no | count | what | color |
|---|---|---|---|---|
| CONTROL-M | 0.888 (98) | 0.973 (73) | 0.938 (65) | 0.950 (20) |
| TREATMENT-M | 0.908 (98) | 0.986 (73) | 0.877 (65) | 1.000 (20) |

## Answers

1. **Does TREATMENT-M preserve the independent GPT support gain while recovering the
   Base-like question distribution?** **Yes.** Supported +0.086 (CI excludes 0),
   statistically identical to CONTROL-M; and the output type distribution is
   Base-like (JS 0.003 from Part E), whereas CONTROL-M collapsed (JS 0.127).
2. **Does Context recover relative to CONTROL-M and the previous bs8 DPO?**
   - vs previous bs8 DPO (Context −0.064) and bs1 final (−0.055): **yes**, half the
     loss (−0.026, CI [−0.042,−0.011]).
   - vs CONTROL-M: **no**; TREATMENT-M Context is −0.029 below CONTROL-M, which held
     Context at Base level (+0.003 ns). So there is a trade: CONTROL-M keeps Context
     but destroys the type distribution; TREATMENT-M restores the type distribution
     at a small Context cost.
