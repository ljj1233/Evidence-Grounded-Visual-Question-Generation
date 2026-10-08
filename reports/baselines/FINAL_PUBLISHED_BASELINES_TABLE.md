# FINAL PUBLISHED BASELINES TABLE

**Not produced on `final_holdout`.** No published baseline (Mostafazadeh 2016
NN-generator, Jain Creativity VAE, LOVA3, Genixer, SeeQ) has frozen-protocol
`final_holdout` predictions in this repository; published-baseline runs were made on
`development1000` only (e.g. `outputs/published_vqg_baselines/`,
`outputs/same_backbone_baselines/seeq_faithful/`). Per the freeze rule, development-set
numbers are **not** substituted into a final-holdout table and no baseline was added
after holdout results became visible.

Therefore this table is intentionally empty. Published baselines must be reported in a
separate category from the internal Full Ours / Full Ours + DPO checkpoints, and only if
their final-holdout results are produced later under the same frozen protocol.
