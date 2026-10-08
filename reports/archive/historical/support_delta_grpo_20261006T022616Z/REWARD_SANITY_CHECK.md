# Reward sanity check

STATUS: **PASS_REWARD_SANITY** (128 train images, 1024 rollouts, frozen pi0)

- supported_gt_unsupported: True
- delta_has_both_signs: True
- not_all_reward_identical: True
- zero_variance_below_0.30: True
- coverage_guard_below_0.50: True
- malformed_below_0.05: True
- template_not_collapsed: True

## Diagnostics

```json
{
  "n_images": 128,
  "n_rollouts": 1024,
  "supported_score_mean": 0.9527950310559006,
  "unsupported_score_mean": -0.7029702970297029,
  "supported_gt_unsupported": true,
  "delta_support_positive": 137,
  "delta_support_negative": 180,
  "delta_support_zero": 707,
  "reward_mean": -0.27639160156250003,
  "reward_std": 0.9679055597401798,
  "mean_group_reward_std": 0.5494024697998636,
  "zero_variance_group_rate": 0.03125,
  "coverage_guard_rate": 0.3779296875,
  "duplicate_penalty_rate": 0.580078125,
  "malformed_rate": 0.0166015625,
  "template_dominance": 0.015625,
  "reward_length_spearman": 0.055582174021886845,
  "q0_support_score_mean": 0.6484375,
  "q0_support_floor_mean": 0.84765625,
  "q0_flat_state_counts": {
    "NO_CONFIRMED_VIOLATION": 103,
    "FAILED": 2,
    "CONFIRMED_VIOLATION": 20,
    "UNCERTAIN": 3
  },
  "ai_flat_state_counts": {
    "CONFIRMED_VIOLATION": 202,
    "NO_CONFIRMED_VIOLATION": 767,
    "FAILED": 17,
    "UNCERTAIN": 38
  },
  "wall_seconds": 727.5149830821902
}
```
