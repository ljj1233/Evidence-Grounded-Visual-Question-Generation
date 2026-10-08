# STAGE G — K=4 within-image diversity statistical audit (no API)

Image-level paired bootstrap, n=10000, seed 42, unit = image. All deltas below have
CIs excluding 0 unless marked ns.

| metric | Base | CONTROL-M | TREATMENT-M | C−B | T−B | T−C |
|---|---:|---:|---:|---|---|---|
| normalized unique@4 | 0.804 | 0.672 | 0.719 | −0.132 [−0.162,−0.101] | −0.085 [−0.109,−0.061] | +0.047 [+0.018,+0.074] |
| unique types@4 | 0.512 | 0.336 | 0.451 | −0.176 [−0.199,−0.152] | −0.061 [−0.076,−0.045] | +0.115 [+0.093,+0.138] |
| within-image exact dup | 0.196 | 0.328 | 0.281 | +0.132 [+0.101,+0.162] | +0.085 [+0.061,+0.109] | −0.047 [−0.074,−0.018] |
| all-four-same-type | 0.199 | 0.660 | 0.340 | +0.461 [+0.395,+0.527] | +0.141 [+0.098,+0.188] | −0.320 [−0.387,−0.250] |

Corpus Distinct-1/2 (not image-level): Base 0.086/0.221, CONTROL-M 0.076/0.191,
TREATMENT-M 0.075/0.192 (no paired CI — no image-level formulation).

## Reading

- The Base→TREATMENT-M residual diversity gap is **statistically real** (all CIs
  exclude 0) — it must not be dismissed as noise.
- TREATMENT-M significantly improves every within-image metric over CONTROL-M.
- Fraction of the Base→CONTROL gap recovered by TREATMENT-M:
  normalized-unique **35.6%**, unique-types **65.3%**, exact-dup **35.6%**,
  all-same-type **69.4%**.
- So matched negatives recover 36–69% of the loss but a significant residual remains.
