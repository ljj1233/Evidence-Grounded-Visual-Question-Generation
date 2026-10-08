# FINAL DPO DELTA TABLE — Full Ours + DPO minus Full Ours (final_holdout)

Paired image-level bootstrap, n=10000, seed 42, 95% CI.

| metric | Full Ours | Full Ours + DPO | paired delta | 95% CI | valid_n |
|---|---:|---:|---:|---|---:|
| GPT Supported | 0.8580 | 0.9365 | **+0.0785** | [+0.0640, +0.0930] | 2000 |
| GPT Unsupported | 0.0445 | 0.0195 | **−0.0250** | [−0.0340, −0.0160] | 2000 |
| GPT Uncertain | 0.0975 | 0.0440 | −0.0535 | [−0.0655, −0.0415] | 2000 |
| SeeQ Coverage | 0.3850 | 0.4244 | **+0.0394** | [+0.0310, +0.0479] | 1995 |
| SeeQ Search | 0.3684 | 0.4166 | **+0.0482** | [+0.0406, +0.0561] | 1995 |
| SeeQ Context | 0.2739 | 0.2296 | **−0.0443** | [−0.0489, −0.0397] | 1995 |
| SeeQ Spatial | 0.0637 | 0.0586 | −0.0050 | [−0.0109, +0.0007] | 1995 |

Local (unpaired, descriptive): BLEU-4 0.2645→0.2595, CIDEr 0.4215→0.4262,
Q-BLEU4-v 0.2558→0.2508; question length 5.85→6.25 tokens; exact duplicate
0.4665→0.5045; type JS(Base→DPO) 0.0497.

No success gate was applied after seeing these numbers; the result is the result.
