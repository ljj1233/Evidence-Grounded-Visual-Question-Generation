# CONFIG — diagnostic run

| item | value |
|---|---|
| seed | 42 |
| GPU | 0 (A800-80GB) |
| git | `9dea27f09aa5c21f4fa53f730923ac1df64169eb` |
| C0 checkpoint | `outputs/ec_qse_aligned_direct4/full_v2/c0/checkpoint-final` |
| C0 adapter sha256 | `f42d399454fff1f0831c23ec6c5b91dbef1b8b5e8763ddffaaaa26f7eafaafaa` |
| E2 checkpoint | `outputs/ec_qse_aligned_direct4/full_v2/e2/checkpoint-final` |
| E2 backbone adapter sha256 | `cafb4929051d2ecdc7f3fe342dbeb691ff7b5ed417c8a41b6770e122b0b4b03f` |
| E2 evidence head sha256 | `17c20aa65b18516c8992569c061c47eb11e8beae649f2f1920f3bdface65e18a` |
| E2 config | N=4, slot_dim=512, λ=(1.0,0.2,0.05,0.01), T=0.07, gate init ≈0.01, out_proj init std 1e-4 |
| training | Final-SFT init, LoRA r16 α32, AdamW lr2e-4, epochs2, batch2×accum8, 1148 steps, seed42 |
| base model | `/hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct` |
| manifest | `reports/e2_slot_use_diagnostic_20261005T022035Z/diagnostic_manifest.jsonl` |
| manifest sha256 | `809e9e0fac0b3f2764bb16925c279c3424895e0c48bd50807cb919c95533485b` |
| manifest size | 256 images (first 256 of development1000 sorted by integer image_id) |
| train overlap | 0 |
| final_holdout | not read; development1000 is disjoint from final_holdout per the frozen split manifests |
| prompt | `Generate a question about this image.` |
| decoding | greedy, max_new_tokens 32 |

No optimizer step, no backward, no parameter change, no external evaluator.
