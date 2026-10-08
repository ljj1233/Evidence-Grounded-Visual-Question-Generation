# Phase 2.6 Code Review

审查对象：`scripts/vvqg/phase26_qdep.py`、`scripts/vvqg/phase26_vfc.py`、`scripts/vvqg/phase26_compare.py`。

## Findings

### P0 — Track A 使用了错误的冻结候选源

`phase26_qdep.py::source_rows()` 读取 `outputs/phase25_offline_reward_audit/candidates4000.jsonl`，而 Phase 2.6 明确要求唯一输入为 `outputs/reward_calibration/sft_candidates_4x.jsonl`。这会使 Track A 与规范的冻结集合不一致。修复：读取 `SOURCE`，并与 Phase 2.5 canon 做键集合一致性检查。

### P0 — QDep token 对齐/评分路径没有可靠的审计保护

实现用裸 `tokenizer.encode(question)` 在多模态 chat template 中搜索问题 token，并计算固定位置。该路径没有明确拒绝 prompt/image/assistant token 混入，且 `prompt` 变量未使用。修复：统一从编码后的完整序列定位 question token span，保存 mask，并增加 debug20 对齐检查。

### P1 — Track B 硬编码模型结构

`LAYER=18`、`HIDDEN=4096`、`num_decoder_layers=36` 没有从冻结 SFT 模型实际配置推导。规范要求先检查实际层数，并固定 `floor(num_decoder_layers / 2)`。修复：从 `AutoConfig` 读取并在 manifest 中记录。

### P1 — VFC 激活缓存破坏了流式边界

`extract()` 将全部 shard 重新 pad/拼接到 `activations_*.pt`，同时训练阶段再一次性读入，违背惰性读取目标并可能制造数 GB 内存峰值。修复：训练阶段直接按 shard 迭代，保留 manifest 作为唯一索引。

### P1 — 支持候选子集评估错误要求每图 4 条

共享评估的 `top_bottom()` 只接受 `len(v) == 4`。supported-only 分析本来会每图留下可变数量候选，导致结果为空或选择性丢图。修复：对每图至少 2 条候选执行 top/bottom，完整四候选要求只用于主评估。

### P1 — GO 与最终决策过于宽松且会自动选择多个信号

决策只比较点估计，未要求 bootstrap CI 支持；脚本还会无条件写入所有信号的 constrained selector。修复：GO 至少要求主指标 top-vs-random 的 95% CI 下界 > 0，并限制 unsupported 不劣于 random 的明确容差；只对唯一获胜信号生成正约束选择器，其他信号仅保留诊断。

### P2 — 比较脚本没有完整产出规范字段

旧 VIC 的表格和相对 random 的 delta 没有统一写入 `signal_bakeoff.json`；需要在 summary 中显式保存 raw SeeQ 分数、top-bottom、concordance、top1 delta 和 unsupported。

## 修复顺序

1. 修正冻结输入与候选键审计。
2. 修正实际模型结构读取和 token 对齐检查。
3. 修正 supported-only 分组与 bootstrap/GO 决策。
4. 加入回归测试并运行 `py_compile`、单元测试和 debug20 级别的无模型协议测试。