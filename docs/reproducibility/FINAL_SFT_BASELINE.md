# Final SFT Baseline

The Phase 2.0 baseline is the Qwen3-VL-8B-Instruct base model with the completed VQAv2 SFT LoRA adapter. The authoritative adapter is /hpc2hdd/home/fye374/LJJ/qg/outputs/checkpoints/qwen3vl8b_lora_r16_vqav2_50k/checkpoint-final; the training summary records base model /hpc2hdd/home/fye374/models/Qwen3-VL-8B-Instruct, 50,000 training records, 2,000 development records, 6,250 steps, two epochs, LoRA rank 16, alpha 32, dropout 0.05, and target modules q/k/v/o/gate/up/down projections. The recorded train loss changed from 3.4093 to 1.1628 and the development loss was 1.3053.

The baseline evaluation uses 1,000 sampled VQAv2 validation images and the active metric registry. Greedy results were BLEU-4 0.317225, CIDEr 0.454971, Q-BLEU4-variant 0.310714, Distinct-2 0.139524, and normalized excess duplicate rate 0.570000.

Reward calibration reuses this frozen image manifest and the existing 1x and 4x generations. The generation prompt was Generate a question about this image. with temperature 0.8, top-p 0.9, and 32 maximum new tokens. The generation seed is 42. The reward-calibration split seed and final-holdout split seed are also 42; 20260926 is not used. Final holdout generation remains untouched.

The training/evaluation protocol is single-image VQAv2 question generation. It is a held-out VQAv2 validation baseline, so the evidence is cross-split within VQAv2 rather than a cross-domain generalization test. Phase 2.1/GRPO is not started by this reconciliation.


Phase 2.3 provenance note: the 1,000-record greedy artifact is the authoritative existing SFT evaluation output and the 4x pool is the existing reward-calibration output. The saved records document prompt, decoding, seed, and token limit, but do not preserve the legacy processor `use_fast`/backend choice, actual batch size, or runtime telemetry. A 16-image canonical zero-shot replay matched the retired wrapper exactly; SFT replay was not used to overwrite the frozen artifact because its legacy preprocessing provenance is incomplete.
