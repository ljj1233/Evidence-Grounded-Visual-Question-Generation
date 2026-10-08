# UVFC Implementation Freeze

This records the implementation audited for Phase 2.6C. No UVFC, SAE, Flat verifier, or SFT checkpoint was changed.

- Qwen target layer: 18 of 36 decoder layers; hidden size 4096.
- Activations: first 16 question-token positions, max pooled over tokens.
- SAE: tied-weight Top-K, dictionary 4096, Top-K 20, ReLU encoder, hard Top-K, zero-scatter decoder, no normalization, AdamW lr 1e-3, 3 epochs, batch 512, seed 42.
- Visual sensitivity: v_j = max(0, p_real(j) - p_blank(j)).
- Undercoverage: u_j = max(0, log((p_anchor(j)+1e-6)/(p_policy(j)+1e-6))).
- UVFC: sum of v_j*u_j over active SAE features.
- Train-only anchors: discovery 20,000; calibration 5,000 real and 5,000 blank; policy 9,184.
- Audit gate: PASS. The current online path needs an additional Qwen forward; the small pilot is feasible.
