# Same-backbone baseline runtime and reproducibility status

## Preflight completed

* seed: `42`
* train records: `50,000`
* unique train images: `9,184`
* development images: `1,000`
* reward-calibration images: `1,000`
* final-holdout images: `2,000` (read only for the zero-overlap assertion)
* train/development, train/reward-calibration and train/final-holdout image-ID
  intersections: `0`
* official Guiding VQG answer-category resource: 16 categories, vendored at
  `src/vvqg/data/baselines/iq_answer_categories.json`
* official repository commits are recorded in
  `docs/SAME_BACKBONE_BASELINES_SPEC.md` and the three YAML configs.

The generated preflight manifest and exact train-record copy are under
`outputs/same_backbone_baselines/manifests/`. No checkpoint, final-holdout
prediction, GRPO run, VIC training, preference data, or violation-reward
training was created.

The existing Qwen3-VL path passed a 16-record/2-step image smoke on an NVIDIA
A800-SXM4-80GB: image-zeroing changed loss by `0.8818`, the loss fell from
`4.1717` to `2.4568`, and peak allocated memory was `18.662 GB`. This validates
the shared image/LoRA path only; it is not a B2/B3/B4 training result.

## Not run

B2 frozen caption/negative/embedding caches, the B2/B3 full two-epoch training,
the B3 detector feature cache, SeeQ R1/R2 proposal/rewrite/filter generation,
and the four 1,000-image evaluations remain unrun. The repository currently has
method-specific entrypoints (`prepare_b2_cache.py`, `train_b2.py`,
`prepare_b3_detector_cache.py`, `train_b3.py`, `run_seeq.py`, and
`train_seeq.py`) but they have not passed their model-backed smoke gates.
Runtime, GPU peak memory, throughput, checkpoint hashes, and output hashes must
be recorded when those runs are made; this document does not invent them.

## Phase 2.4C implementation status

Shared trainer and method-specific infrastructure now exist under
`src/vvqg/training/` and `src/vvqg/baselines/`, with focused regression tests
covering target-token masking, B2 projection gradients and frozen caches, B3
selector gradients, and resumable SeeQ stages. The current gate state is in
`outputs/same_backbone_baselines/readiness.json`. Full caches, model-backed
smoke gates, training, and evaluation remain pending because GPU 0 is occupied
by an existing job; no running process was interrupted and no full result is
claimed.

## CPU thread oversubscription vs the container CPU quota (vLLM)

This was the largest runtime problem in Phase 2.4C and is now fixed.

Root cause:

* The container is capped at **8 CPUs** by the cgroup:
  `/sys/fs/cgroup/cpu/cpu.cfs_quota_us = 800000`,
  `cpu.cfs_period_us = 100000`. Both files are `root:root 644` and are **not
  writable from inside the container**.
* PyTorch/OpenMP size their thread pools from the *host* core count (64+),
  ignoring the cgroup quota. vLLM's client-side multimodal preprocessing (the
  HF image processor) runs torch/torchvision CPU ops, so it launched 8+
  threads for very small tensors.
* The vLLM client and EngineCore together demanded more than 8 CPUs, so the
  cgroup throttled them. `cpu.stat` showed `nr_throttled` / `throttled_time`
  climbing monotonically, and the *same* PIL resize measured anywhere from
  `2 ms` to `700 ms` across calls.

Symptom: SeeQ R1 generation ran at `~256` records/min with GPU utilization
around `7%`; the vLLM client process burned `~5` CPU cores doing OpenMP
spin/oversubscription rather than useful work.

Fix, in `VLLMQwen3VLBackend.__init__` immediately before `LLM(...)` is built:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)
```

The EngineCore subprocess inherits the environment, so both processes are
covered. This is applied only on the vLLM path; the HF backend still needs its
CPU threads and is left multi-threaded. No image processor was changed, so the
generated pixel values (and therefore resumable stage files) are unaffected.

Measured effect (A800-SXM4-80GB, SeeQ R1, `--batch-size 128`):

* throughput: `256` -> `2176` records/min (`~8.5x`)
* vLLM client CPU: `~5` cores -> `0.25` core; GPU utilization rose to `~85%`
* over a 60 s window with the fix active: `nr_throttled` and
  `throttled_time` both `+0` (throttling stopped)
* batch scaling for the propose stage (fresh images, 256 new tokens):
  `64 reqs = 42 req/s`, `128 reqs = 76 req/s`, `160 reqs = 79 req/s`; batch
  `128` is used.

`nr_throttled` is cumulative since container start and cannot be reset from
inside the container; the useful signal is whether it still grows. Raising the
8-CPU quota itself requires the container administrator (`docker --cpus`), but
it is unnecessary once CPU threads are pinned to 1.
