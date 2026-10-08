# 当前方法技术说明：Full Ours 与独立 DPO refinement

2026-10-08。定义对象：`outputs/phase5c_qse_relativeguard/checkpoints/step_final`（step256）及其 DPO 后处理 `reports/offline_dpo_unified_refinement_20261008/dpo/checkpoints/step_00207`。以下区分代码事实、历史运行事实与未解决的溯源项；不将当前源码自动当成历史源码快照。

## 1. 术语、正式位置与输入输出

| 术语 | 英文名称及确定性 | 正式定义 / 核心函数 | 输入 → 输出 | 最终 loss 中的作用 | 最终推理是否需要 |
|---|---|---|---|---|---|
| VFC | **Visual Feature Coverage**，`phase26_vfc.py` 开头明确写 visual feature coverage | [VFC 适配说明](VFC_FAC_ADAPTATION_SPEC.md)；[uvfc.py](../../src/vvqg/rewards/visual_information/uvfc.py)::`TopKSAE/_question_pooled_sae/score_uvfc` | image + question + frozen scorer/SAE/v → scalar VFC | G2 记录诊断值；排序实际用 UVFC | 否，训练准备/训练评分用 |
| UVFC | **Undercoverage-weighted Visual Feature Coverage** 是按代码公式给出的描述性展开；已检索源码和现有定义文档，**未找到该英文全称的正式命名声明**，论文采用前应确认名称 | [UVFC 冻结定义](UVFC_IMPLEMENTATION_FREEZE.md)；[phase26_vfc.py](../../scripts/vvqg/phase26_vfc.py)::`calibrate`；`uvfc.py::score_uvfc` | image + question + frozen scorer/SAE/v/u → scalar UVFC、VFC、active feature IDs、question token count | 离线选 teacher；在线作为 violation-first 排序第二键，再转 rank advantage；非直接可微 loss 项 | 否 |
| QSE | **Question Space Expansion**；完整阶段称 **Evidence-Constrained Question Space Expansion**，见 [Phase5C 定义](PHASE5C_EVIDENCE_CONSTRAINED_QSE.md) | [phase5b_direct4_generate.py](../../scripts/vvqg/phase5b_direct4_generate.py)::`main`；[phase5c_select.py](../../scripts/vvqg/phase5c_select.py)::`s3`；[phase5c_build_targets.py](../../scripts/vvqg/phase5c_build_targets.py)::`main` | image → K4 candidates → UVFC/Flat signals → selected q* → D1 targets | q* 进入 D1 CE，得到 G2 初始化；G2 loss 没有独立 QSE term | 否；候选采样用模型推理，但属于训练准备，不是最终部署路径 |
| RelativeGuard | 现有标题为 **Relative degeneration guard**；实现称 **initialization-relative guard**，不是另一个缩略词或独立模型类 | [online_uvfc_relative_grpo.py](../../src/vvqg/training/online_uvfc_relative_grpo.py) 模块说明、`_guard_baseline/_question_stats/_degenerate/run_online` | frozen SFT/D1 校准统计 + 当前训练统计 → thresholds、bool、stop_reason | 控制何时停止和保存；不加到 reward/loss，不反传 | 否 |

没有 `QSE` 或 `RelativeGuard` 同名 class。不能用新造类名代替这些实际函数。VFC/UVFC 是 SAE 特征代理分数，不等于外部 SeeQ 的 Visual Evidence Coverage 指标。

## 2. UVFC：激活、SAE、频率与公式

### 2.1 激活提取

来源：`uvfc.py` L17–23、L57–64、L75–110；离线构建对应 `phase26_vfc.py::Activations.batch`。

1. 输入 RGB image 和 question；组成 user image + `Generate a question about this image.`、assistant question 的 chat template。
2. 冻结 Qwen 前向，取 `hidden_states[LAYER + 1]`，`LAYER=18`、hidden size=4096。精确定义是隐藏状态数组索引 **19**，避免论文“第18层”产生索引歧义。
3. 用 `tokenizer.encode(question, add_special_tokens=False)` 在完整序列中匹配 question token span，仅取前 `min(16, question_length)` 个位置；不池化 image/prompt tokens。
4. 每个 token 的 float32 隐状态进入 tied-weight Top-K SAE；先稀疏编码，再对问题 token 维度取 max；padding 位置在 pooling 前 mask 为 0。
5. pooled activation `>0` 得到二值 active set。因此是独特激活特征加权和，不是激活幅值和、问题 token 均值或图像 patch 覆盖率。

### 2.2 SAE

令 token 隐状态 \(h_t\in\mathbb R^{4096}\)，\(W\in\mathbb R^{4096\times4096}\)，\(b\in\mathbb R^{4096}\)：

\[
z_t=\operatorname{TopK}_{20}\!\left(\operatorname{ReLU}((h_t-b)W)\right),
\qquad \hat h_t=z_tW^\top+b.
\]

`TopKSAE.forward` 使用 `relu → torch.topk → zeros_like.scatter → sparse @ weight.T + bias`。字典 4096、TopK20、无归一化。SAE 的独立训练目标是 \(\operatorname{mean}(\hat h-h)^2\)，AdamW lr=1e-3、3 epochs、batch512、seed42；见 `phase26_vfc.py::train_sae` L250–274。其输入实现是缓存 activation tensor 的 flatten（包含缓存 padding 位置），不是最终 G2 loss 的一部分。

\[
a_j(I,q)=\mathbf1\!\left[\max_{t\le\min(16,|q|)}z_{t,j}>0\right].
\]

### 2.3 冻结的视觉敏感度与 undercoverage

`phase26_vfc.py::train_split/calibrate` 与 [manifest.json](../reproducibility/EXTERNAL_REFERENCE_INDEX.md#external-27ea0c50d912)：从 SFT train50k 用 seed42 拆出 discovery20k、calibration5k。calibration 的同一问题分别配真实图和相同尺寸 RGB(127,127,127) 灰图。频率按**问题级二值 activation**统计：

\[
p_{\mathcal D}(j)=|\mathcal D|^{-1}\sum_{(I,q)\in\mathcal D}a_j(I,q),
\quad v_j=[p_{real}(j)-p_{blank}(j)]_+,
\]
\[
u_j=\left[\log\frac{p_{anchor}(j)+10^{-6}}{p_{policy}(j)+10^{-6}}\right]_+.
\]

- \(p_{anchor}\)：discovery20k 的人工训练问题。
- \(p_{policy}\)：Final SFT 对全部 9184 个唯一训练图像各生成一个 greedy 问题的激活频率。
- SAE、\(v,u\) 构建时的激活模型都是 Final SFT；不使用最终 holdout、GPT 支持标签或 SeeQ 分数训练这些权重。
- 磁盘冻结资产：`outputs/phase26/vfc/sae.pt`、`weights.pt`；后者包含 v、u、p_anchor、p_policy、eps 等。

### 2.4 VFC / UVFC

\[
\mathrm{VFC}(I,q)=\sum_{j=1}^{4096}a_j(I,q)v_j,
\qquad
\mathrm{UVFC}(I,q)=\sum_{j=1}^{4096}a_j(I,q)v_ju_j.
\]

精确代码：`uvfc.py::score_uvfc` L118–133，`active=pooled>0`；`(active*v).sum` 与 `(active*(v*u)).sum`。输出 list 每条含 `UVFC/VFC/question_token_count/active_features/_active`。

**必须披露的模型差别**：teacher scoring 的 `phase5b_score_candidates.py` 默认加载 Final SFT；G2 的 `run_online` L261–263 创建的 reference 是 **D1**，L364–365 将这个 reference 传入 `score_uvfc`。所以 G2 使用 D1 隐状态 + SFT 训练的冻结 SAE/v/u。没有证据表明 G2 重建了 D1 专属 SAE。`variant=static`、`refresh_every=0`；当前策略不会刷新 u。不能把历史 Dynamic UVFC 的 `dynamic_u()` 写入最终 G2 算法。

## 3. QSE：扩展、约束筛选、D1 蒸馏

### 3.1 候选扩展

`phase5b_direct4_generate.py` 的真实 DIRECT4 prompt 为：

```text
Generate one natural question about this image.
The question must be answerable from the image.
Output only the question.
```

Final SFT、K=4、T=0.8、top_p=0.9（`generate_questions` 默认）、max_new_tokens=64、seed42、sdpa。代码在每张图的 4 个输出中剔除空串及完全相同的重复字符串，保留生成顺序。

\[
\mathcal C(I)=\operatorname{dedup}\{q_k\sim\pi_{SFT}(\cdot\mid I,p_{Direct4})\}_{k=1}^4.
\]

训练 teacher pool 实际有 35,097 条 / 9184 图。DIRECT4 与 D1/G2/最终推理的单行 instruction **不同**；这是现有实现，不应在论文中误写全程 prompt 完全一致。DPO 后处理才使用 aligned 候选池。

### 3.2 Flat 与 S3

`StructuredFlatVerifier.verify_structured` 复用冻结 Qwen2.5-VL-7B Flat prompt/parser。其依赖链：`structured_flat.py → online_flat.py → frozen_verifier.py → calibration.py`，prompt 文件是 [visual_support_prompts.json](../../configs/new_vqg/visual_support_prompts.json)。

`support_summary` 将各 requirement 标签映射为：supported/clear/answerable=1，uncertain/ambiguous=0.5，unsupported/unanswerable=0。`support_floor` 是所有已识别 requirement 分值的 **最小值**；parser failure 为 -1，无 requirement 为 0。

\[
s(I,q)=\min_r\operatorname{score}(label_r),\qquad
\mathcal E(I)=\{q\in\mathcal C(I):state(q)\ne\mathrm{CONFIRMED\_VIOLATION},\ s(I,q)\ge0.5\}.
\]

\[
q^*(I)=\begin{cases}
\arg\max_{q\in\mathcal E(I)}\mathrm{UVFC}(I,q),&\mathcal E(I)\ne\varnothing,\\
S1(\mathcal C(I)),&\text{otherwise}.
\end{cases}
\]

`phase5c_select.s3` L26–31：相同 UVFC 用 `sha256("42:image_id#generation_id")` 升序稳定打破并列。fallback `phase5b_select.s1` L33–39 按 `(is_confirmed_violation, -support_floor, -UVFC, tie_hash)` 排序。

此约束允许 UNCERTAIN；“非确认违例”不等于“确认 supported”。空可行集的 S1 fallback 不保证满足 τ 或无违例。τ=0.5 的校准说明在 Phase5C 文档 §B，使用冻结 Phase2.1 dev+val 800 标签；不能把“UVFC 权重 train-only”扩大成“所有方法选择完全未用开发集”。

### 3.3 D1 loss

`phase5c_build_targets.main` 对 10k manifest 每条记录写 `{image, q_epoch1:q*, q_epoch2:q*}`；图像重复记录保留，因此 D1 不是 image-balanced 9184 条训练。D2 multipos 文件是并行实验，未进入 G2 初始化。

`phase5c_distill.py` L112–136 → `prepare_rows/make_collator` → `model(**batch).loss`。设 \(\mathcal T_{assistant}\) 为 prompt 和 padding 被 mask 后的 assistant 序列位置：

\[
L_{D1}(\theta)=-\frac{1}{|\mathcal T_{assistant}|}\sum_{t\in\mathcal T_{assistant}}
\log\pi_\theta(y_t\mid I,p_{student},y_{<t}).
\]

该 CE masking 是 prompt/pad=-100；assistant chat 结束 token 若在 full template 中保留也会参与 CE。不要将它和 RL/DPO 的仅 question-token span 掩码混为一谈。D1 无 target weight 字段，权重默认1；两 epoch、effective batch16、1250 steps、lr2e-4、warmup37 steps + cosine。QSE 通过这些 targets 训练 D1；**G2 没有 QSE 辅助 loss，也没有 SFT replay 项。**

## 4. G2 最终总 loss 与真实更新行为

### 4.1 采样与 advantage

实际配置：[qse_relativeguard.yaml](../../configs/phase5c/qse_relativeguard.yaml)，运行快照：[config.json](../reproducibility/EXTERNAL_REFERENCE_INDEX.md#external-9602ab158101)。

- init/reference=D1；原 SFT 只另用于 guard calibration 和冻结资产来源。
- 每步8条 image records ×4 samples；T0.8、top_p0.9、max32，seed 随 step 确定。
- 每个问题先得到 D1-space UVFC，再由 `OnlineFlatVerifier.verify` 得到 flat_state。
- `canonicalize_group(members,"p2")` 首先按是否 confirmed violation，再按 UVFC 降序排序。FAILED/UNCERTAIN 在此非 violation 分支；不像 S3 那样施加 support_floor≥0.5。
- `norm_question` 合并大小写/空白等价的重复标签语义，重复样本本身仍在组中；exact duplicate 的 UVFC 必须一致。同排序键共享占据名次的平均 reward。
- rank slots 为 `(1.5,0.5,-0.5,-1.5)`，不是直接把 UVFC 当 reward 标量。

\[
r_i=\operatorname{mean}(\text{occupied rank slots}),\quad
A_i=\frac{r_i-\bar r}{\sqrt{\frac14\sum_j(r_j-\bar r)^2}+10^{-8}}.
\]

`fixed_group_relative.py::canonicalize_group/assign_tie_rewards/_mean_std` L75–182；组长度不等4、存在短于2字符问题或 rank 零方差时跳过。训练 manifest 的人工 question 不进入 RL reward 或 loss。

### 4.2 Token loss

`policy_logprobs.encode_group → encode_items → gather_token_logprobs` 只取 question token 对应 logprob，预测位置是 `start-1 ... start+count-2`；prompt/image/padding/chat-ending token 不计入此 loss。令 \(T_i\) 是候选问题 token 数：

\[
\rho_{it}=\exp\left(\operatorname{clamp}(\ell^\theta_{it}-\operatorname{stopgrad}(\ell^{old}_{it}),-20,20)\right),
\]
\[
k_{it}=\exp(\ell^{ref}_{it}-\ell^\theta_{it})-(\ell^{ref}_{it}-\ell^\theta_{it})-1,
\]
\[
\boxed{L_{G2}=\frac1{|\mathcal B_{usable}|}\sum_{g\in\mathcal B_{usable}}\frac14\sum_{i=1}^4
\frac1{T_i}\sum_{t=1}^{T_i}
\left[-\min\big(\rho_{it}A_i,\operatorname{clip}(\rho_{it},0.8,1.2)A_i\big)+0.02\,k_{it}\right].}
\]

精确位置：`online_uvfc_relative_grpo._group_terms` L152–175；组平均梯度 `(loss/len(usable)).backward()` L418。先平均每条响应的 token，再平均组成员，再平均可用组，因此不是全 batch token 一次统一加权。

**总 loss 只有 policy surrogate + KL。** 无独立 VFC loss、QSE loss、RelativeGuard penalty、entropy bonus、latent-coverage loss、DPO loss或价值网络。Flat 通过排序进入 advantage，不是数值惩罚项。

### 4.3 此 checkpoint 实际不是多轮 PPO 更新

`run_online` L401–428 在每组更新前即时计算当前 policy 的 detached old logprobs，再计算 current logprobs；所有组 backward 完毕后才 optimizer.step。`inner_epochs=1`，且 `build_trainable_model` 将 policy 置 eval 模式（梯度仍启用）。因此 old/current 权重相同，ratio≈1，clipping 没有实际起作用。

本次读取全部训练日志：ratio_mean 范围 `[0.9999999403953552,1.0]`，最大 clip_fraction=0。old detach 后梯度仍非零；实际梯度行为相当于带组标准化 rank advantage 的 on-policy policy gradient + KL anchor。论文可以陈述 clipped-surrogate 代码，但不能声称该运行进行多轮 PPO ratio clipping。后续 G3 corrected-PPO 是另一支，不是本 checkpoint。

优化器 AdamW lr1e-6、betas(0.9,0.999)、wd0、grad norm clip1；cosine 计划1000 updates。只训练 LoRA（43,646,976 参数）；bf16、sdpa。

## 5. RelativeGuard：条件、实际停止与版本差异

在 seed42 打乱的 train10k 前32条记录上，各采样4问，分别测原 Final SFT 的 \(D_{SFT0}\) 和初始化 D1 的 \(D_{init}\)。重复率在这128条问题上全局计算；训练监控在每步32条 rollout 上全局计算，**不是同图内部 K4 重复率**。malformed 定义仅为 strip 后长度<2。

\[
T_{dup}=D_{init,dup}+0.2,\quad
T_{len}=D_{init,len}+0.5D_{SFT0,len},\quad
T_{bad}=D_{init,bad}+0.02.
\]

\[
degen_s=(dup_s>T_{dup})\lor(len_s>T_{len})\lor(bad_s>T_{bad}).
\]

实际 [training_summary.json](../reproducibility/EXTERNAL_REFERENCE_INDEX.md#external-e815e7035e52)：

| 项目 | SFT0 | D1 init | 最终阈值 |
|---|---:|---:|---:|
| duplicate_rate | 0.1015625 | 0.2578125 | 0.4578125 |
| mean token length | 6.765625 | 6.640625 | 10.0234375 |
| malformed_rate | 0 | 0 | 0.02 |

当前源码还有：loss/reference_kl/grad_norm 非有限即停止；reference KL>0.2 连续10个记录停止；clip_fraction>0.5 连续10个记录停止；`DEGEN_STREAK=3` 时最近3个记录各自 degen 为真才停止（可由不同退化指标触发）。检查在 optimizer update 之后。

**历史 checkpoint：** CHECKPOINT.global_step=256，summary.stop_reason=DEGENERATION。最后三步 dup=`0.15625,0.1875,0.46875`；长度和 malformed 都未越界。全日志用当前 `_degenerate` 回放只有 step256 单步为真，没有三步连续触发点。

[phase5c_followup/REVIEW.md](../reproducibility/EXTERNAL_REFERENCE_INDEX.md#external-3feefa87a423) §2 记录当时 patience=none；[coverage iteration recommendation](../../reports/archive/diagnostics/coverage_iteration_20261004T023610Z/FINAL_RECOMMENDATION.md) 记录之后加入 patience=3，但 G2 1000-step rerun **未启动**。故：历史单步停止行为有日志/报告支持；当前代码是后续修正版；历史精确代码快照和源码 hash 仍未知。不得把 patience3 写成 step256 的已证实训练条件。

停止后 `save_checkpoint(...,"step_final")` 保存触发停止后的当前参数；代码没有回滚到前一步或按 dev 选 best 的机制。

## 6. Training pipeline 与 inference pipeline

训练分阶段进行，并非一个端到端联合总目标：

```text
VQAv2 train50k {image, question}
  → Final SFT (CE)
  → 分支A：SAE 训练 + real/blank/anchor/policy 频率 → 冻结 v/u
  → 分支B：Direct-4 → UVFC + Structured Flat → S3(tau=.5)
  → targets_d1_single → D1 (assistant CE)
  → G2: image → current-policy K4 → frozen D1 UVFC + Online Flat
      → p2 rank + advantage → policy surrogate + KL(D1) → LoRA update
      → RelativeGuard continue/stop → Full Ours step_final (256)
```

最终推理：Qwen3-VL-8B public base + **一个最终 LoRA adapter**，PIL image processing +固定指令 → `model.generate` → question。Final SFT/D1/G2 是 adapter 参数继承关系，部署时不依次叠加三个 adapter。DPO 模型同理，直接加载 step207 adapter。

| 生命周期 | 组件 |
|---|---|
| 训练准备/训练专用 | SAE/v/u calibration、UVFC scorer、Flat verifier、Direct-4 teacher、S3、D1 CE、冻结 reference、rank advantage、RelativeGuard、DPO pairs/reference/loss |
| 最终推理 | public base + final adapter、processor/tokenizer、固定 image-only VQG instruction、自回归解码 |
| 推理后的评估 | BLEU/CIDEr/Q-BLEU4-v、SeeQ-Luna、GPT-4o support；不反馈改变已冻结模型 |

通用入口 `scripts/vvqg/generate.py`；已有 G2 内置推理 `_generate` 与 checkpoint sweep `phase5a_generate_checkpoints.py` 都调用同一 `inference.generate_questions` 家族。最终 holdout accounting 显示双方 greedy、max32、seed42、batch16；旧 Markdown freeze 写 batch8，实际 JSON 已解释双方同批次16。此处只读取记录，未重新生成。

## 7. DPO refinement：独立后处理

### 7.1 数据构建与 matched-negative

实际来源链：

```text
Final SFT + aligned instruction + train9184 / K4
 → ec_aligned_direct4_generate.py (35139 candidates)
 → dpo_v1_formal_score.py (UVFC + Flat)
 → dpo_v1_formal_pairs.py (6633 original pairs)
 → overnight_matched_negative.py (2733 replacements)
 → unified merge (persisted builder not located; all-row lineage verified)
 → UNIFIED_PAIRS.jsonl → pairs_train.jsonl (6633)
```

原始 pair builder 过滤：overall_visual_support ∈{supported,unsupported}、flat_state 非 FAILED/None、非空短串、同图 norm_question 只出现一次。chosen=NO_CONFIRMED_VIOLATION 中 UVFC 最大，rejected=CONFIRMED_VIOLATION 中 UVFC 最小；每图一对。同分按原输入顺序（Python max/min）取，未另用 QSE tie hash。

matched negative：固定 chosen；同一图寻找 confirmed violation 且 overall unsupported、与 chosen 同 broad question type、长度≥2且归一化问题唯一的候选。按以下 key 升序选：

```text
(是否不同 interrogative head,
 abs(len(tokens(rejected))-len(tokens(chosen))),
 round(-TFIDF_cosine,6),
 rejected_question_string)
```

`tokens` 来自 `vvqg.evaluation.question_types`，不是 Qwen token count。TF-IDF unigram+bigram、sublinear_tf，fit 于训练候选池。2733 图能匹配，其余3900图保留原负例；chosen 全部不变。

源码细节：`overnight_matched_negative.py` 的输出字段 `chosen_UVFC` 实际取了 `rej.get("UVFC")`，字段名与值来源不符。最终 trainer 输入只有 image_id/image/chosen/rejected，未消费该字段，所以不改变本次追溯的 DPO loss。

`dpo_v1_formal_union.py` 只构建评估 queue。最终6633对 unified 合并没有在当前 scripts/src/configs 中定位到专用持久化程序；[DATA_MANIFEST.json](../reproducibility/EXTERNAL_REFERENCE_INDEX.md#external-3fba4b502431) + 原始/matched/unified/trainer 全行比对确认上述规则，但不能声称找到完整历史 launcher。

### 7.2 DPO loss

入口 `scripts/vvqg/dpo_v1_formal_train.py::main` → `src/vvqg/dpo_pilot/trainer.py::run_dpo`。`_seq_logprobs` L47–66 使用 `encode_items/gather_token_logprobs`，对 question token logprobs **求和**，无长度归一化。

令 \(\ell_\theta(I,q)=\sum_t\log\pi_\theta(q_t|I,p,q_{<t})\)，reference 为冻结 Full Ours G2：

\[
\boxed{L_{DPO}=-\frac1B\sum_{i=1}^B\log\sigma\!\left(0.1\left[
\ell_\theta(I_i,q_i^+)-\ell_{ref}(I_i,q_i^+)
-\ell_\theta(I_i,q_i^-)+\ell_{ref}(I_i,q_i^-)
\right]\right).}
\]

代码 `trainer.py::run_dpo` 的 margins/logsigmoid 分支。虽有可选 weighted 分支，最终 pairs_train 无 weight，实际取普通 batch mean。没有额外 GRPO/UVFC loss，没有 RelativeGuard 调用；导入 relative trainer 是复用 load/save/RNG helper。

实际 resolved config：Full Ours 同时为 init/reference，LoRA r16/alpha32/dropout0.05，lr5e-6、AdamW、constant LR、beta0.1、batch8、seed42、1 epoch、830 steps。训练确实运行到了830；最终选择 **已有 trajectory 中 step207**，不是训练总共只运行207步。step207暴露1656 pair records，原训练集仍为6633对。

### 7.3 step207 selection 与 eval

- DEV256：`reports/offline_dpo_checkpoint_trajectory_20261008/DEV256_MANIFEST.jsonl`；`PHASE1/PHASE1_REPORT.md` 比较207/414/621/830。207相对后期 checkpoint 有较小 type JS、较好 K4多样性，较小 BLEU/CIDEr损失，并保留 preference learning。
- 这并不等于207在所有指标都超过 Full Ours：报告的 K4 norm-unique 为0.704，Full Ours为0.804；Context 也下降。选择是记录中的折中规则，不能转述为全指标获胜。
- `reports/offline_dpo_freeze_20261008/FREEZE_RECORD.json` 固定step207；DEV744 secondary confirmation 是 formal dev1000 减DEV256，不是从未使用过的新 test set。
- `reports/final_evaluation_20261008/FINAL_HOLDOUT_REPORT.md` 已有最终 holdout 结果。因此早期 FREEZE_RECORD 的“holdout remains sealed”仅为冻结时刻状态。
- 未定位专门实现“自动选step207”的函数；现有证据为轨迹表、selection report 与 freeze record 的人工/离线决策记录。`generate.py`、`overnight_analyze.py`、`overnight_paired_stats.py`、canonical evaluators 提供相应生成/统计能力。

## 8. 本次核验与剩余缺口

| 核验 | 结果 |
|---|---|
| 现有 S3/s1/tie_hash 函数纯 CPU 回放全部 D1 targets | 10000条完全一致 |
| original→matched/fallback→unified→trainer payload | 6633条完全一致；2733 matched +3900 fallback；chosen不变 |
| current `_degenerate` 对全 G2 log 回放 | 仅step256单次越界；没有连续3次触发 |
| G2 ratio / clipping 日志 | ratio_mean≈1；max clip_fraction=0 |
| 模型 forward / SAE 重算 / 新评测 | 未执行；本任务禁止训练，且代码溯源无需这些运行 |

剩余溯源项：历史 G2 精确源码快照；unified merge 的持久化 builder/历史 shell 命令；UVFC 正式英文全称声明；部分临时评估 orchestration 的完整启动命令。它们被明确标注，未用文件名或默认值补全为已证实事实。
