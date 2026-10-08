# Reward source audit

Actual formula (from `src/vvqg/support_delta/reward.py` and `trainer._reward_groups`):

```
R_i = delta_support_i - 0.50*coverage_guard_i - 0.25*duplicate_penalty_i - 0.50*malformed_penalty_i
delta_support_i = S(q_i) - S(q_0)
A_i = (R_i - mean_group(R)) / std_group(R)   (0 if std <= 1e-8)
```

No clipping, rank transform, filtering, cross-image normalization, or component-wise standardization is applied to the reward; only the group standardization above.

Data source: frozen-pi0 sanity rollout (128 groups / 1024 candidates); trained step300 per-candidate rollouts were NOT saved.

Key numbers: support-label source = Flat verifier flat_state/support_floor (same-source grouping, NOT external validity); delta_support nonzero rate = 0.3096; WITH_VS_WITHOUT_S0_MAX_ADVANTAGE_DIFF = 5.773e-15.
