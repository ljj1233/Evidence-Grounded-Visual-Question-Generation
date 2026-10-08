# CHECKPOINT-ONLY cleanup — 2026-10-08

状态：已执行并完成验证。只删除确认废弃 checkpoint 的模型/训练状态二进制；没有删除 checkpoint 目录，也没有归档或移动其他文件。

## 最终摘要

```text
CHECKPOINTS_DELETED: 153
SPACE_FREED: 56.61 GiB payload (60,782,161,334 bytes)
CHECKPOINTS_RETAINED: 191 (188 local checkpoints + 3 external published/vision models)
FINAL_SFT_PRESENT: YES
D1_PRESENT: YES
FULL_OURS_PRESENT: YES
DPO_STEP207_PRESENT: YES
ACTIVE_GQA_PRESENT: YES
ACTIVE_BASELINES_PRESENT: YES
TEST_RESULT: PASS: 155 passed, 1 existing warning; imports PASS
```

## 删除范围及计量

- 删除 153 个 checkpoint 的 278 个二进制文件，分属 26 个实验根目录。
- 删除文件逻辑字节数：60,782,161,334 B（56.6078 GiB）。
- 删除文件原占用块：60,801,025,024 B（56.6254 GiB，按 `st_blocks × 512`，所有删除文件 `st_nlink=1`）。这是本次解除链接文件的占用量；未用共享文件系统的全局 `df` 差值冒充本任务精确回收量。
- `size_bytes` 与 payload SHA256 均针对实际删除的权重/状态文件；tokenizer/config 等留存文件不计入回收空间。
- `DELETE_LOG.tsv` 每行对应一个删除文件，记录原路径、字节数、实际 SHA256、步数、删除原因、替代模型、报告路径。
- 每个 checkpoint 目录写入 `CHECKPOINT_REMOVED.md`（153 个），每个实验根目录另有聚合说明（26 个）。

删除文件类型：

| 文件名 | 数量 |
|---|---:|
| `adapter_model.safetensors` | 153 |
| `evidence_head.pt` | 4 |
| `optimizer.pt` | 1 |
| `rng_state.pth` | 1 |
| `scheduler.pt` | 1 |
| `training_state.pt` | 118 |

## 判定依据与删除前门槛

- `docs/HISTORICAL_PHASE_MAP.md`：旧训练权重与最终 SFT → D1 → G2 → DPO 的真实继承关系。
- `docs/SCRIPT_CLASSIFICATION.tsv`：CURRENT/UNKNOWN 入口、配置及其保留依赖；同时检查 src、tests 和上次归档依赖表。扫描记录见 `SOURCE_REFERENCE_SCAN.tsv`。
- `reports/repository_cleanup_audit_20261008/DPO_CHECKPOINT_AUDIT.tsv`：历史 DPO 清单。旧审计把各 run 的 `step_final` 都标记为 final；本次依用户明确的旧终点删除范围、阶段沿革和 `reports/offline_dpo_freeze_20261008/FREEZE_RECORD.md` 判定。当前正式冻结模型是 step207，旧终点标签不能作为当前模型身份。
- 全部删除项具有已保存的配置、summary/report、指标（含训练 loss/step 记录）、明确的 global step、逐文件 SHA256；已有 CHECKPOINT.json hash 均核对一致。
- Phase4A replay 的步数来自原 train_log.jsonl 最后一条实际 `global_step=250`，并由保留的 manifest 指向 checkpoint；没有把计划 max_steps 当作实际完成步数。
- 7 个 Phase4A `checkpoint_final` 是同一历史实验内的目录别名。确认 CURRENT/UNKNOWN/test 不加载这些别名；目录与别名均保留，原位置说明明确权重已删除。详见 `HISTORICAL_ALIAS_REVIEW.json`。
- 删除前检查文件 inode/mtime/size 与 hash 时一致，确认待删 payload 未被进程打开；未发现对应实验运行命令。`EXECUTION_GATE.json` 保存实际执行计划 SHA256。
- 通用加载器接受显式 CLI checkpoint 参数，不构成对全部历史 checkpoint 的当前直接依赖。旧实验重新推理需要其已删除权重；当前主方法与活动 GQA/baseline 的加载路径保留。

## 分实验删除统计

| 原 experiment directory | checkpoint 数量 | 删除 GiB |
|---|---:|---:|
| `outputs/ec_qse_aligned_direct4` | 8 | 1.436 |
| `outputs/phase3a_fixed_group_relative` | 11 | 1.789 |
| `outputs/phase3a_multiseed_p1strong` | 6 | 0.976 |
| `outputs/phase4a2_rft_replay` | 1 | 0.163 |
| `outputs/phase4a3_rft_replay_multiseed` | 2 | 0.325 |
| `outputs/phase4a_uvfc_rft` | 2 | 0.325 |
| `outputs/phase4a_uvfc_rft_multiseed` | 2 | 0.325 |
| `outputs/phase4b_online_uvfc/base_uvfc` | 2 | 0.651 |
| `outputs/phase4b_online_uvfc/base_uvfc_baseanchor` | 4 | 1.628 |
| `outputs/phase4b_online_uvfc/dynamic_uvfc` | 6 | 2.604 |
| `outputs/phase4b_online_uvfc/feasibility` | 4 | 1.628 |
| `outputs/phase4b_online_uvfc/static_uvfc` | 6 | 2.604 |
| `outputs/phase5a_static_uvfc_latentcov` | 7 | 3.092 |
| `outputs/phase5b_explore_ground_distill` | 2 | 0.651 |
| `outputs/phase5b_explore_ground_uvfc` | 3 | 1.139 |
| `outputs/phase5c_distill_multipos` | 1 | 0.163 |
| `outputs/phase5c_static_relativeguard` | 8 | 3.581 |
| `reports/evidence_constrained_full_20261005T165813Z` | 2 | 0.651 |
| `reports/g3_support_preserving_rl_20261005T105701Z` | 10 | 4.232 |
| `reports/offline_dpo_bs8_controlled_20261008` | 10 | 4.557 |
| `reports/offline_dpo_matched_negative_20261008` | 12 | 5.208 |
| `reports/offline_dpo_pilot_20261006T180506Z` | 5 | 2.116 |
| `reports/offline_dpo_unified_refinement_20261008` | 4 | 1.953 |
| `reports/offline_dpo_v1_formal_20261007T102440Z` | 8 | 3.581 |
| `reports/risk_aware_dpo_relaxed_guard_fresh256_20261007T051622Z` | 4 | 1.628 |
| `reports/support_delta_grpo_20261006T022616Z` | 23 | 9.602 |

Unified DPO：删除实际 global_step **414 / 621 / 828 / 830**；830 对应 `step_final`。**step207 与 step0000 保留**。所有 trajectory、freeze、model/hash、final evaluation 记录原样保留。

VC-QSE：在本次 outputs/reports 权重扫描中未发现独立 adapter/model checkpoint；没有据名称扩展删除候选包、嵌入或其他 `.pt` 数据。

## 保留项

- 本地 188 个 checkpoint 全部保留原二进制；另验证 3 个外部 published/vision model（LOVA3、Genixer、CLIP）。详见 `RETAINED_CHECKPOINTS.tsv`。
- 四个指定祖先：Final SFT、D1、Full Ours step256、unified DPO step207，全部 adapter SHA256 与删除前一致。
- GQA checkpoint 根目录 43 个、baseline checkpoint 根目录 151 个全部保留；这两类有交集，不能相加当作总数。
- 保留全部 outputs/checkpoints 下的 SFT/旧 SFT 快照、Full Ours 中间快照、UVFC SAE/weights、活动 GQA/baseline 及其所有中间 checkpoint；未扩大到这些目录。
- 不确定项保留如下：

| 原路径 | 保留原因 |
|---|---|
| `outputs/phase4b_online_uvfc/base_uvfc/init_adapter` | UNKNOWN_REQUIRES_REVIEW: initialization snapshot not needed for priority cleanup |
| `reports/g3_followup_20261005T161303Z/pilot2/checkpoints/step_0000` | UNKNOWN_REQUIRES_REVIEW: incomplete run records |
| `reports/joint_soft_evidence_qse_20261004T104324Z/smoke/smoke_e1` | UNKNOWN_REQUIRES_REVIEW: global_step unavailable |
| `reports/joint_soft_evidence_v2_20261004T135351Z/smoke/smoke_e2` | UNKNOWN_REQUIRES_REVIEW: global_step unavailable |

- 除明确列出的删除 payload 外，保留所有 config、summary、report、metrics、manifest、hash/provenance、trajectory、tokenizer/processor 和实验数据。未删训练参数配置 `training_args.bin`。
- `.venv*`、`data/`、代码/scripts/configs/docs、当前 `.git`、外部模型目录未执行任何写入/移动/删除命令。未安装依赖、修改环境、启动训练或干预任务。

## 验证结果

- 原快照中 **44,135** 个非删除文件 inode/size/mtime/mode/link count/blocks 未变化；额外缺失 0，意外变化 0。
- 278 个计划 payload 均已不存在；每条删除记录与计划吻合；所有保留 checkpoint payload 存在。
- 四个祖先：原 adapter SHA256 不变；adapter_config 可解析，base model 目录存在；safetensors header、tensor offsets 与文件长度一致。
- 当前 GQA base-init D1/G2、GQA SFT 加载路径存在；所有扫描到的 GQA/baseline checkpoint 与外部 released model 文件保留。
- CURRENT_MAIN_PIPELINE 导入：15 个脚本 + 10 个核心模块通过。
- CURRENT_EVAL 导入：22 个脚本通过。
- 删除前 pytest：155 passed, 1 warning in 13.08s。
- 删除后 pytest：155 passed, 1 warning in 25.98s。既有 warning 位于 tests/test_trainer_base.py:37（requires_grad tensor 转标量）。
- 测试通过进程内 sys.path 添加 src，并禁用 bytecode/cacheprovider；未修改项目配置或持久环境。
- 没有实例化大模型、跑生成或进行 GPU 推理；加载验证范围为路径/配置/基座目录/权重 hash/序列化结构和 Python 导入。

测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-vllm-cu128-v011/bin/python -B -c 'import sys; sys.path.insert(0,"src"); import pytest; raise SystemExit(pytest.main(["tests/","-q","-p","no:cacheprovider"]))'
```

## 文件索引

- [DELETE_CHECKPOINT_PLAN.tsv](FINAL_REPORT.md)：删除前完整判定计划，包含保留项。
- [DELETE_LOG.tsv](DELETE_LOG.tsv)：实际逐文件删除记录。
- [RETAINED_CHECKPOINTS.tsv](RETAINED_CHECKPOINTS.tsv)：保留权重与理由。
- [VERIFICATION.json](FINAL_REPORT.md)：路径、文件快照、祖先 hash 与结构验证。
- [PYTEST_AFTER.log](../../EXTERNAL_REFERENCE_INDEX.md#external-927ff712f8ff)、[IMPORT_MAIN.log](../../EXTERNAL_REFERENCE_INDEX.md#external-e532bd278455)、[IMPORT_EVAL.log](../../EXTERNAL_REFERENCE_INDEX.md#external-8432642b55bb)。

STOP.
