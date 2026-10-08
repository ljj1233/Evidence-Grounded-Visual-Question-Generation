# REWARD DEFINITION — G3 conservative support-preserving RL

Per image, K=4 rollout questions. For each rollout:

```
R = 1.00 * support_continuous
  + 0.20 * coverage_anchor
  - 0.25 * duplicate_penalty
  - 0.15 * generic_penalty
```

## support_continuous (weight 1.00)
```
support_continuous = clip(2 * (support_floor - 0.5), -1, 1)
if flat_state == CONFIRMED_VIOLATION: support_continuous -= 0.5
if malformed / parser_failure / empty / FAILED: support_continuous = -1.0
```
`support_floor` is the frozen local Flat verifier's weakest requirement score
(0 / 0.5 / 1.0) from `structured_flat.support_summary`. No external API.

## coverage_anchor (weight 0.20)
`UVFC` from the frozen Static-UVFC scorer, rank-normalised **within the image's 4
rollouts** to `[-1, 1]` (`2*rank/(K-1) - 1`). It is a Coverage-keeping term only;
UVFC is never the sole reward.

## duplicate_penalty (weight -0.25)
1.0 for a rollout whose normalised question equals another rollout of the **same
image**; 0 otherwise. Cross-image equal questions are not duplicates.

## generic_penalty (weight -0.15)
Deterministic local rules (no LLM judge): flag when the question is <=2 words,
matches a fixed generic template (`what is in the picture`, `what is this`,
`is this a picture/photo/image`, `what do you see`, `describe the image`,
`what color is the image`, ...), or has no content word outside a small
stopword/generic set (`image, picture, photo, thing, object, scene, color,
see, ...`).

## Advantage
Group-relative: `adv_i = (R_i - mean(R_group)) / (std(R_group) + 1e-6)`; groups
with `std(R) < 1e-6` are skipped.

## Implementation (corrected PPO)
- `old_logprob`: computed **once at rollout time** (before any optimizer step),
  detached, stored on CPU.
- `new_logprob`: current grad-bearing policy.
- `ratio = exp(clamp(new - old, -20, 20))`, reused across `inner_epochs=2`.
- PPO clipped surrogate, `clip_epsilon=0.2`.
- KL anchor = **fixed initial Full Ours checkpoint** (`step_final`), `beta_kl=0.02`.
- `lr=5e-7`, AdamW, `max_grad_norm=1.0`, cosine schedule.
