# Fixed-Rollout Group-Relative Policy Optimization Pilot — Method

Status: Phase 3A pilot. This document describes the objective and data contract.
Results and runtime are in `docs/FIXED_GROUP_RELATIVE_PILOT_RESULTS.md` and
`docs/FIXED_GROUP_RELATIVE_PILOT_RUNTIME.md`.

## 1. What this is, and what it is not

The four completions per image are **fixed rollouts** generated once by the
frozen Final SFT behavior policy (`temperature=0.8`, `top_p=0.9`,
`max_new_tokens=32`, 2,000 SFT-train images, seed 42). The policy does **not**
regenerate responses after each update. Therefore this run is a
**Fixed-Rollout Group-Relative Policy Optimization** pilot. It is **not standard
on-policy GRPO**, not "TRL GRPO", and not an on-policy loop. No value model,
critic, reward model, or GAE is used: the group-relative advantage comes
directly from the frozen within-group ranking.

## 2. Behavior and reference policy

Behavior policy `pi_old` = frozen Final SFT = reference policy `pi_ref`. Both are
the `Qwen3-VL-8B-Instruct` base plus the Final SFT LoRA adapter
(`outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final`, r=16,
alpha=32, dropout=0.05). Because `pi_old == pi_ref`, one frozen cache
(`old_policy_cache/`) serves both roles.

Per-response behavior token log-probabilities are precomputed with the **same
per-group microbatch** (one image, four questions) used during training. bf16
logits are not invariant to batch shape, so this alignment is required for
step-0 parity: without it, two numerically different forward passes of the same
completion can differ by ~0.2 nats purely from GEMM tiling.

The exact generation prompt is reconstructed with the Qwen3-VL chat template
(`add_generation_prompt=True`) and the frozen instruction; the completion span is
the contiguous question tokens after the prompt (odd/even token-boundary cases
raise). `old_policy_cache/` stores, per response: `completion_start`,
`completion_token_ids` (the completion mask content), `completion_count`, and the
per-token `old_logprobs`. Prompt/image tokens are never scored; padding is masked.
Only question tokens are used (`completion_count > 0`); the template EOS/newline
is excluded because the raw generated token stream was not retained, so EOS
presence cannot be confirmed.

## 3. Group-relative advantages

For group `g` (one image, four candidates):

- **P1** orders candidates by descending UVFC.
- **P2** orders by violation-first lexicographic rule: admissible candidates
  (`NO_CONFIRMED_VIOLATION`, `UNCERTAIN`, and `FAILED`) before
  `CONFIRMED_VIOLATION`; within each partition, descending UVFC.

Rank-slot base rewards are `(+1.5, +0.5, -0.5, -1.5)`. Equal sort keys form a
tie block and receive the **mean** of the rank slots the block occupies, so
identical or equally-scored candidates never receive contradictory signals.
Exact duplicate questions are verified to share a UVFC value (tolerance 1e-6);
`CONFIRMED_VIOLATION` is treated as not admissible. `FAILED` flat states are
canonicalized to a consistent successful state when a duplicate copy carries one,
otherwise kept as `FAILED` and logged.

Advantages are the population-standardized rewards within the group:
`A_i = (r_i - mean(r)) / (std(r) + 1e-8)`. Groups with `std(r) <= 1e-8` are
zero-variance, mark `zero_variance_group = true`, and are excluded.

## 4. Objective

Only completion (question) tokens are scored; prompt and padding tokens are
masked out. For response `i` and completion token `t`:

```
log_ratio[i,t] = clamp(logp_theta[i,t] - logp_old[i,t], -20, 20)
ratio[i,t]     = exp(log_ratio[i,t])
surrogate[i,t] = min(ratio*A_i, clip(ratio, 1-0.2, 1+0.2)*A_i)
```

The surrogate is averaged over completion tokens **per response**, then averaged
equally over responses, so longer responses do not carry more weight. The policy
loss is `-mean(response_surrogate)`.

A reference-deviation penalty is added with `beta_KL = 0.02`:

```
ref_log_ratio[i,t] = logp_ref[i,t] - logp_theta[i,t]
kl_token[i,t]      = exp(ref_log_ratio) - ref_log_ratio - 1   (>= 0)
total_loss         = policy_loss + 0.02 * mean(kl_token)
```

Because the samples are fixed behavior-policy rollouts, this term acts as a
**reference-policy deviation penalty**; it should not be overclaimed as exact
on-policy KL.

## 5. Optimization

AdamW, learning rate `1e-6`, cosine schedule with 5% warmup, gradient clipping
`1.0`, one epoch, microbatch of one complete group (four responses), gradient
accumulation of eight groups (32 responses per optimizer step, ~250 steps).
The trainable copy is kept in `model.eval()` (dropout off) while gradients stay
enabled, so importance ratios contain no dropout noise. Only the copied LoRA
adapter is trainable; the base model is frozen. P1 and P2 start from independent
copies of the exact same Final SFT state and the same data order (group-level
shuffle, seed 42).

## 6. Why this bridge

Offline selection already ranks the frozen rollouts well. This pilot is a
**feasibility bridge** between successful offline selection and a future
on-policy policy-optimization stage: can the offline UVFC / violation ranking be
learned back into the SFT policy distribution? A successful result is evidence
that fixed-rollout group-relative learning can transfer the ranking signal. It
does **not** establish on-policy GRPO success, and no larger or on-policy run is
launched automatically.
