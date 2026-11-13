# Evidence-Grounded Visual Question Generation

教育图文问题生成：输入一段文本与配套图像，生成一道需要联合使用材料信息的问题。

**当前主线：先分析baseline生成的问题是否“可回答且图文缺一不可”，再决定方法实验。** 可答性/忠实性与必要性分开，实际答题表现及BDR另列。

- **全1257组ScienceQA baseline（gpt-6-luna，plain-v1）已生成、回答并评估完成**，目录`outputs/luna-api-baseline-test1257-*`；原评估器自动暂定64题双必要（5.1%）。
- **新版必要性评估器**（`necessity-ensemble-*`，两个模型组加仲裁，答案槽位判等价）已并入`src/`，在全1257组上自动暂定38题（3.0%），`outputs/necessity-v3-full-v1/`。人工抽查：38题中确认双必要19、偶然不变性7、误放行12，另发现漏判4；已核实双必要下限约23/1257（1.8%）。口径、限制和误差来源见该目录`probe_report.md`第八、九节。
- **报告口径只有两类**：已核实双必要与其余（待裁定并入其余，仅作诊断）。
- 尚未做：对所提方法的公平比较、非教育数据（WebQA、MultiModalQA）上的泛化检验、整体重复运行评估稳定性。

## 文档入口

| 文档 | 作用 | 状态 |
|---|---|---|
| [VQG_Proposal.md](VQG_Proposal.md) | 研究定义：任务、指标、评估与实验边界 | 研究定义以此为准 |
| [story.md](story.md) | 论文叙事：研究问题、章节骨架、拟贡献 | 随研究进展修改 |
| [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md) | **数学参考**：双模态必要性的定义、命题与证明；第9节为评估实现约定 | 当前 |
| [docs/references.md](docs/references.md) | **论文与数据参考**索引，附数学命题索引 | 当前 |
| [docs/evaluation_protocol_v1.0.md](docs/evaluation_protocol_v1.0.md) | 早期评估协议（五维判据、固定分母） | 已冻结，不修改 |
| [docs/evaluation_design_v2_draft.md](docs/evaluation_design_v2_draft.md) | 早期评估设计草案（英文） | 历史，已被上面数学参考与评估器取代 |
| [docs/implementation_plan.md](docs/implementation_plan.md) | 实施路线、阶段验收与命令 | 当前，较长 |
| [PROGRESS.md](PROGRESS.md) | 唯一进度日志：完成项、验证、产物、下一步 | 追加式 |
| [AGENTS.md](AGENTS.md) | 供编码助手遵循的项目约定 | 当前 |

## 环境与离线检查

项目使用一个 `.venv/` 和 `uv.lock`。默认不调用真实 API、不下载数据、不训练模型。

```bash
uv sync
.venv/bin/python -m pytest
```

若本机 `uv` 默认缓存目录不可写，可先指定可写缓存，例如：

```bash
UV_CACHE_DIR=/tmp/opencode/uv-cache uv sync
```

## 信息必要性评估入口（推荐）

两个模型组（默认`gpt-6-luna`与`DeepSeek-V4.1-Flash`，网关模型名大小写须一致）各自做单模态充分性筛查和反例复算，仲裁模型（默认`gpt-5.6-terra`）只处理两组没拿定的题，所有判定由代码按固定规则给出；答案等价由仅从题干生成的答案槽位决定。定义与约定见[数学参考第9节](docs/necessity_formal_argument.md)。需先有完成的`simple-baseline-eval`运行作为来源。

```bash
E=".venv/bin/python -m egvqg --run-id necessity-v3-full-v1"
$E necessity-ensemble-init --source-run luna-api-baseline-test1257-eval-v1   # 冻结题目、提示、模型与上限；同配置幂等
$E necessity-ensemble-run --limit 10      # 先小批量真实预检；去掉--limit全量，--phase screen|arbitrate|all
$E necessity-ensemble-report              # 固定分母报告；离线，写results.json与final_items.json
```

- 并发默认32，每次调用至多重试一次，HTTP硬上限默认16000；失败的调用留在`results.jsonl`但不缓存，恢复时补跑。
- 报告只分“确认双必要”与“未确认”；确认项必须人工复核后才计入，评估器自身会误放行（对称方格、被预设绑定的交换题等）。
- 只有原评估器给出两个方向假想情形的题才可能被确认双必要；其余只能被否定或保持未决。

## 全test API baseline入口

最新执行分成两步，生成器只见hint与原始材料图像，不见数据集原Q/选项/答案/solution。每个请求独立无历史；不改写生成失败，不按评估结果换题。全部1257题始终保留。

```bash
# 生成与完整图文回答已完成；查询其记录和用量
.venv/bin/python -m egvqg simple-baseline-report --run-id luna-api-baseline-test1257-simple-v1
# 全批完成后才创建的独立评估，现已全部完成
.venv/bin/python -m egvqg simple-baseline-eval-report --run-id luna-api-baseline-test1257-eval-v1
```

生成/回答run含`questions.jsonl`、`answers.jsonl`和全部`records.jsonl`，按attempt_id/question_sha256绑定；`execution_acceptance.json`记录实际输入重建与16路并发验证，`image_profile.json`记录图片限制。没有在该阶段构造参考、跑四条件或评分。配置入口为`simple-baseline-init`，默认`--concurrency 16 --max-image-edge 1024 --max-image-pixels 1000000 --max-image-bytes 524288`；每题至多2次新HTTP，总上限2514。完全相同的旧输入与有效模型配置才能复用，失败也保留，不重计旧消费。

评估入口为`simple-baseline-eval-init/run/report`。init会拒绝尚未全部完成的生成/回答批次。复用冻结K及现有open/verify-v2，一题至多5次评估HTTP，全run上限6285，默认并发16；先检查可答性和原答案支持，再检查模态必要性，发现任一模态可省略即停止该题剩余方向，不把未检查方向填成否。不会启动四条件或计算新的BDR；既有历史指标不改。API各角色同模型，结果暂定自动，需内容与证据抽查，不能视为独立人类金标准。恢复同一run时仅补缺失记录；运行中不另开重复任务。

## 本地baseline的test评估入口

忠实性复评只读取已有Q/T/I/K与具名评审JSONL，不重跑生成、回答或API；新run保留固定分母和原始来源哈希。检查预设、答案支持、所求范围；参考错误与问题错误分开，确定生成缺陷才令V=0。通用审查模板为`prompts/review-question-faithfulness-v1.txt`，此次模板未发送至模型。

```bash
.venv/bin/python -m egvqg faithfulness-review \
  --source-run qwen3-vl8b-baseline-test30-v1 \
  --run-id qwen3-vl8b-baseline-test30-separated-v1 \
  --reviews outputs/qwen3-vl8b-baseline-test30-faithfulness-v1/reviews.jsonl
```

同命令仅校验并恢复既有派生复评；变更评审须新run_id。结果不覆盖历史协议、默认流水线或旧评分。

当前已完成30组官方test随机样本的baseline分析，排除此前3组接入材料，仅plain-v1自然生成，不预筛联合求解空间。Qwen3-VL-8B负责生成和四条件作答，gpt-6-luna负责新Q参考、评分与现有v2必要性分析/核验；Codex具名AI复核。先报告固定30题中可回答且双必要的确认数量，再分别列可回答性/忠实性和双模态必要性；必要性待裁定不等于不能回答。实际答题表现与行为BDR另报。

本轮按用户指定复用 `/home/lijunjian/video_agent/video_agent_skill_learn/Video-skill-transfer/.venv`，已有Torch 2.8.0 CUDA 12.8、Transformers 4.57.6及Accelerate 1.10.1，只补装python-dotenv。项目 `.venv` 继续用于离线检查。实际推理环境与权重哈希写入运行记录；没有新建环境或复制模型权重。

```bash
# 已完成的30个baseline固定名额；dry-run只检查待执行步骤
.venv/bin/python -m egvqg local-experiment-run \
  --run-id qwen3-vl8b-baseline-test30-v1 --model-path ~/models/Qwen3-VL-8B-Instruct \
  --device cuda:3 --max-local-calls 150 --max-requests 160 \
  --stages generation reference --dry-run
```

真实执行时，将命令前缀改为 `PYTHONPATH=src /home/lijunjian/video_agent/video_agent_skill_learn/Video-skill-transfer/.venv/bin/python` 并移除 `--dry-run`；先完成具名参考审核，再运行 `--stages answers scores`。本地调用和HTTP调用分别累计限额，恢复复用已保存输出。必要性沿用 `configs/necessity_assessment_v2.json` 和现有 verify-v2，不采用保留约束分离诊断的新模板。需要独立重建项目推理依赖时可用 `uv sync --extra local-inference`；本轮实际推理版本以指定共享环境快照为准。

## 仅评估信息必要性的最小入口

真实原题修复版本使用`configs/necessity_assessment_v2.json`，分析与verify共享K与冻结参数。最终离线汇总必须指定verify运行；`results.json`仅是原证明复核，合并结果在`verified_results.json`。复核来源为已知参考的Codex委托AI，不冒充独立人类金标准。完整执行命令见[实施方案](docs/implementation_plan.md#真实原题评估的修复入口dev-v2)。

按用户选择，后续默认网关模型改为`gpt-6-luna`，模型名通过`LLM_JUDGE_MODEL_NAME`设置。已有20题结果仍属于`gpt-5.6-luna`，不改写或合并为新模型结果；切换模型使用新run_id，缓存按模型身份隔离。同一方法比较批次保持评价模型一致。工程推进不以模型审核100%无误为门槛，继续保留固定分母、未决、已知偏差和针对性复核。

```bash
.venv/bin/python -m egvqg open-necessity-run --run-id real-scienceqa20-v2 --max-requests 44
.venv/bin/python -m egvqg open-necessity-report --run-id real-scienceqa20-v2 \
  --verification-run real-scienceqa20-verify-v2
```

证据verify侧车已实现并测试：5份正确证据全保留、5份明确破坏证据全拒绝；两份花架边界仍被宽解释放行，所以不宣称全自动稳定。verify应位于证据生成后、D_T/D_I确认前；失败只使证据未决，不翻转标签，原20题未改。报告`outputs/necessity-verify-dev-v1/probe_report.md`。

```bash
.venv/bin/python -m egvqg verify-open-evidence --source-run open-necessity-dev20-v1 \
  --run-id necessity-verify-dev-v1 --items data/processed/necessity-verify-dev-v1/items.jsonl --max-requests 16
```

开放取值扩样已完成：10组新材料、20个新问题，沿用旧模型/Prompt，40次分析原始标签与预期一致；严格复核38份证据支持、2份花架证据边界未决。18题双向完整确认；核心“双必要”8是/11否/1未决，不能把标签一致称为100%可靠。报告`outputs/open-necessity-dev20-v1/probe_report.md`，问题入口同目录`questions.html`。

```bash
.venv/bin/python -m egvqg open-necessity-run --run-id open-necessity-dev20-v1 --max-requests 44
.venv/bin/python -m egvqg open-necessity-report --run-id open-necessity-dev20-v1
```

开放取值探针已执行：4组日常图文、8道构造问题，不给候选取值、参考答案或答案组件。16次分析提出12份答案变化见证、4份充分解法，均经Codex逐项复核；仍为已知预期的开发控制，非自然题通用准确率。问题入口`outputs/open-necessity-dev-v1/questions.html`，结果与限制在`outputs/open-necessity-dev-v1/probe_report.md`。

```bash
.venv/bin/python -m egvqg open-necessity-run --run-id open-necessity-dev-v1 --max-requests 20
.venv/bin/python -m egvqg open-necessity-report --run-id open-necessity-dev-v1
```

答案可辨识性小探针已执行：11道dev题（8新合成控制、1已有自然生成Q、2已暴露控制题），22次真实模型分析的答案集合均与预声明有限情形的程序穷举一致，无解析失败。支持“检查最终答案是否仍唯一”的可行性信号，尚非开放域准确率。完整边界与成本见`outputs/identifiability-dev-v1/probe_report.md`。

```bash
# 已完成运行重放0新调用；report为离线重算
.venv/bin/python -m egvqg identifiability-run --run-id identifiability-dev-v1 --max-requests 26
.venv/bin/python -m egvqg identifiability-report --run-id identifiability-dev-v1
```

后续两项验证已完成：`evaluator-necessity-norm1`零调用派生复评确认17项基线证据可用，两个已知坏输出仍被拒；`evaluator-necessity-probe2`在13题上执行26次计划分析（24次HTTP、2次缓存），14个分析单位存在格式问题，v3未通过单次合同验证。主报告：`outputs/evaluator-necessity-probe2/evaluation_report.md`。旧结果未覆盖，当前不推荐直接扩批。

```bash
# 零调用派生复评；完成后复用，且派生run禁止发送模型请求
.venv/bin/python -m egvqg necessity-normalize --source-run evaluator-necessity-probe1 --run-id evaluator-necessity-norm1
# v3已完成：恢复新增0步骤
.venv/bin/python -m egvqg necessity-run --run-id evaluator-necessity-probe2 --max-requests 30
```

`necessity-*`只评估已有T/I/Q，不生成问题，不运行BDR。每组件分别检查无模态解法或两份合法补全，确认结论须有具名证据审核。预期标签不进入模型输入，未决保留在固定N。首轮语义复核来源为Codex委托AI，非独立人类金标准。

```bash
# 已完成运行：恢复为0新步骤；预算不能超过冻结的24次
.venv/bin/python -m egvqg necessity-run --run-id evaluator-necessity-probe1 --max-requests 24
# 无凭据离线重算
.venv/bin/python -m egvqg necessity-report --run-id evaluator-necessity-probe1
```

实现位于`src/egvqg/necessity.py`，规则位于`metrics.py`；独立配置`configs/evaluation_v2.json`。`necessity-init`导入已审核参考，`necessity-review`导入绑定具体分析哈希的证据审核。首轮报告明确区分原始标签、结构合格提案和复核结论；当前不适合直接批量评价。

## 新版固定名额实验入口

新版协议为`three-phase-prompt-v1`。所有指标使用预安排名额N作分母；自动、人工、最终裁定分别汇总，旧运行继续使用原协议。完整字段、状态、公式、人工接力与命令见[实施方案第6节](docs/implementation_plan.md#6-新版固定名额工程入口three-phase-prompt-v1)。

```bash
# 同配置恢复，只补pending步骤；完成运行不会再发请求
.venv/bin/python -m egvqg experiment-run --run-id prompt-dev-smoke --max-requests 24

# 独立离线计算，无需API凭据
.venv/bin/python -m egvqg records-metrics --records outputs/blindspot-12-v1/records.jsonl

# 固定种子60组候选，无模型调用；同run_id复用原抽样
.venv/bin/python -m egvqg prepare-admission --run-id scienceqa-admission-60-v1 --limit 60 --seed 20261112
```

- `outputs/repeatability-3x5/`：3个固定dev问题各5次新HTTP请求，载荷与原始输出字符级一致；仅称所测输入上的“等效贪心重复性通过”，不宣称后台机制已验证。
- `outputs/blindspot-12-v1/`：12题最终委托裁定、交叉表、`delegated_review.jsonl`、`disagreements.md`及5个典型案例；最终裁定在adjudicated，human字段不冒填，协议与结果哈希已冻结。
- `outputs/scienceqa-admission-60-v1/`：首60及补抽72组的原图审阅页、132条准入理由、冻结的`materials_final_30.jsonl`。按原顺序取前30个完全通过项，30通过/15待审/87排除；未参考生成结果。
- `prompts/*-v1.txt`：独立维护的生成、证据、答题、审核和评分模板，`$变量`占位符缺失即报错。

### 阶段二一键运行与恢复

```bash
.venv/bin/python -m egvqg stage-two \
  --run-id scienceqa-admission-60-v1-plan \
  --evaluation-run blindspot-12-v1 \
  --materials outputs/scienceqa-admission-60-v1/materials_final_30.jsonl \
  --repeatability-run repeatability-3x5 --method both --max-requests 1000
```

同命令恢复，已完成名额跳过；也可用`--method plain`或`strong`，strong要求plain30全部终态。`--dry-run`校验冻结契约并估算预算，不发送请求。主记录120条，阶段二只完成其中60条；锚定与消融仍pending。阶段二报告为`stage_two_summary.md`，方法比较均使用N=30；`calibration/report.md`为每方法8题的委托复核，`stage_two_cases.md`解释案例与自动误判，`acceptance.json`保留240输入隔离与零请求重跑验收；`stage_two/`导出预安排的60条及其原计划。显式`retry-evaluation`只重评失败的指定评测步骤，不替换生成问题。

### 分阶段、参考答案审核门控的最小探针

最新修复探针为`outputs/anchor-dependency-dev4-v4/`：同4组dev材料上的三轮提示调试已完成，见`probe_report.md`和`acceptance.json`。最终四题来源复核均有效、按冻结规则均需T/I；遗传/电池的单模态拒答，气体/溶液仍有默认假设命中，BDR为0/4、未决2/4。此前只问50%的对称遗传题不应判T必要，更正单独保留。不是持出集或总体方法增益。

新模板通过初始化参数选择，旧v1和正式运行保持原样：

```bash
.venv/bin/python -m egvqg experiment-init \
  --run-id anchor-dependency-dev4-v4 \
  --materials outputs/anchor-headroom-dev4-v1/materials.jsonl \
  --methods anchored --partition dev \
  --config configs/experiments_review_gated.json \
  --prompt-override pair-v1=pair-v4 \
  --prompt-override generate-anchored-v1=generate-anchored-v2 \
  --prompt-override extract-image-v1=extract-image-v2
```

该run已完成generation/reference及审核后的answers/scores，勿覆盖旧Q。新材料应新建run_id并重新审核；自动标签、委托标签及行为BDR分别报告。

`anchor-headroom-dev4-v1`已由Codex完成8题来源复核和6题四条件评测，另2题参考未决保留；共63次请求。完整结果见`outputs/anchor-headroom-dev4-v1/diagnostic_report.md`，复核与验收分别在`source_review/`和`probe_acceptance.json`。下列为流程示例；恢复实际运行时仅选已核验有效名额，不对未决题启动答题。

新配置`configs/experiments_review_gated.json`只用于新运行，不与`configs/experiments.json`下的结果合并。提示、模型接口与评价协议v1.0相同；差别是每题一次候选参考答案、审核决定后才评分，以及有记录的JSON反斜杠修复。

```bash
RUN=anchor-headroom-dev4-v1   # 已初始化：4组dev材料 × strong/anchored = 8名额
# 1) 预算预估，不调用模型
.venv/bin/python -m egvqg experiment-run --run-id $RUN --dry-run
# 2) 只生成；先看Q与配对证据（records.jsonl的generation字段）
.venv/bin/python -m egvqg experiment-run --run-id $RUN --stages generation --max-requests 130
# 3) 候选参考答案、四条件作答、相关性；随后停在awaiting_reference_review
.venv/bin/python -m egvqg experiment-run --run-id $RUN --max-requests 130
# 4) 审核者填写outputs/$RUN/reference_review_template.jsonl的decision/note/reviewer/source后导入
.venv/bin/python -m egvqg reference-review --run-id $RUN --reviews outputs/$RUN/reference_review_template.jsonl
# 5) 只补评分与依赖参考的审核；已完成步骤不重发
.venv/bin/python -m egvqg experiment-run --run-id $RUN --max-requests 130
# 仅来源盲审包（匿名ID、T、原图路径、Q及哈希；不含方法、候选参考、自动结果），只生成一次
.venv/bin/python -m egvqg review-packet --run-id $RUN
# 只跑指定名额与阶段；其余名额保持pending并留在分母
.venv/bin/python -m egvqg experiment-run --run-id $RUN --attempt-ids <IDS> --stages answers scores --max-requests 130
# 只读查看四条件结果模式（C对/W错/R拒答/U未评分）
.venv/bin/python -m egvqg records-patterns --records outputs/$RUN/records.jsonl --attempts
```

`decision`取`accept_candidate`、`corrected`（另填answer与rubric）、`unresolved`或`invalid_question`；`source`为`human`，或带`user_authorized`与`authorization`的`user_delegated_ai`。未填decision的行保持待审。`--max-requests`是该运行跨恢复累计的HTTP上限。

## 阶段 A smoke

用 `tests/fixtures/stage_a/` 中的 6 组合成材料跑通：

```text
图文材料 → 生成 Q → 独立候选答案与核验 → 有效性与四条件作答 → 评分 → 汇总
```

```bash
.venv/bin/python -m egvqg smoke
.venv/bin/python -m egvqg smoke --run-id stage-a-small --limit 2
.venv/bin/python -m egvqg summarize --run-id stage-a-smoke
```

- 第一次运行写入 `outputs/<run_id>/`；相同 `run_id` 和相同配置再次运行会恢复并跳过已有记录，不重复计数。
- 改变配置、提示或材料后应使用新的 `run_id`，否则会拒绝覆盖。
- 主要产物：`questions.jsonl`、`generation_attempts.jsonl`、`validity.jsonl`、`answers.jsonl`、`responses.jsonl`、`scores.jsonl`、`calls.jsonl`、`metrics.json`、`summary.md`。`calls.jsonl` 保留每次调用的输入审计、原始输出和用量；人工复核的记录格式已在 `src/egvqg/records.py` 定义，阶段 A 不生成人工标签。
- 三种生成方法：`plain`、`strong`、`evidence_anchored`。证据锚定流程包含文本证据提取、图像证据提取、跨模态配对和出题。
- 技术失败按 `run.max_attempts` 有限重试；解析失败或语义生成失败不静默重试，分别记录为 `generation_failure` 与 `technical_failure`。
- 指标严格使用预先安排的尝试数 N 作固定分母；无效题和生成/技术失败保留；未决值单列，不按答错处理；无材料作答正确率使用“有效且答案已核验且仅 Q 条件可评分”的分母。

## ScienceQA 下载与候选导入

```bash
.venv/bin/python -m egvqg download-data --split train --limit 3
# 相同命令复用已下载文件；也可完全离线重新导入
.venv/bin/python -m egvqg download-data --split train --limit 3 --offline
```

采用 [ScienceQA 官方下载源](https://github.com/lupantech/ScienceQA#-download-the-dataset)，元数据固定到提交 `2cbf8318e07b9ece895bb2ae605e71e38d623264`。首次下载约 30 MiB 元数据和约 600 MiB 训练图像包；`--limit` 限制导入材料数，不限制官方 ZIP 下载大小。`--max-download-mb` 限制图像包大小（默认 1024 MiB），下载最多尝试 3 次、支持 `.part` 续传，完成文件通过 SHA256 校验后复用。

- 原始数据及来源/校验收据：`data/raw/scienceqa/<revision>/`，只读复用。
- 默认产物：`data/processed/scienceqa/hint-image-v1/train-n3-seed0/`，包含 `materials.jsonl`、全训练分区的 `selection.jsonl` 与 `manifest.json`。
- 默认仅使用非空 `hint` 与原始图像；原题、选项、原答案、`lecture`、`solution` 不进入材料。`--seed` 控制固定随机抽样；图像哈希作为共享图像分组键，图像按路径引用。
- 本次训练候选池 3,944 条，导入 3 条，全部 `review_status=pending`。非空图文只是规则初筛，不能据此认定具备双模态必要性；跨分区来源/模板去重和人工审查仍待完成。

## MultiModalQA / WebQA 原始数据下载

```bash
.venv/bin/python -m egvqg fetch-dataset --dataset multimodalqa --workers 2
.venv/bin/python -m egvqg fetch-dataset --dataset webqa --workers 8 \
  --proxy http://127.0.0.1:7890 --image-mirror
```

`fetch-dataset`只下载/解压/校验，不筛材料或调用模型。原始文件保留在`data/raw/<dataset>/<upstream-revision>/`，来源URL、文件大小、SHA256和记录数量写入manifest及下载收据；`.part`可恢复。WebQA的`--image-mirror`使用固定Hugging Face提交的图像分卷，经hf-mirror传输并按LFS SHA256验证；官方Google Drive来源仍保留，标注/索引取自官方发布。`--proxy`只用于本下载流程的Drive/公开文件访问，不改变模型网关配置。

来源：[AllenAI MultiModalQA](https://github.com/allenai/multimodalqa)、[WebQnA WebQA](https://github.com/WebQnA/WebQA)。两者均已完整下载、解压并校验，`manifest.json`均为`status=complete`：

| 数据集 | 题目分区 | 原始证据 | 本地目录 |
|---|---|---|---|
| MultiModalQA | train23,817 / dev2,441 / test3,660 | 文本218,285、表10,042、图像记录57,058 | `data/raw/multimodalqa/4dd14328c6d02a4daa357cc6032915a0b14602e3/` |
| WebQA | train36,766 / val4,966 / test7,540 | 图像记录389,750；正文/事实保存在原标注JSON | `data/raw/webqa/81cc377aaddb2cf9de98cf082a2795ce53ccc1f4/` |

MultiModalQA原图在`final_dataset_images/`，保留6个gz JSONL和原ZIP，约占4.6GiB。WebQA标注在`annotations/`，原始图片以base64保存在`image_store/imgs.tsv`（72,234,224,104字节），用根目录`imgs.lineidx`索引；51个原始分卷也保留，总占用约119GiB。按官方规则`int(image_id) % 10000000`逐一核对全部分区图像引用，389,750个唯一引用ID均对应正确记录，其中test涉及90,760个唯一图像ID。

WebQA尾部慢卷通过有界小范围请求补齐，整卷SHA256和7z解压校验均通过；不是只下载前50卷。解包使用官方7-Zip 25.01缓存工具，包SHA256固定在代码中。来源清单、实际下载URL、镜像提交、文件大小/哈希和校验时间均在raw目录的manifest/收据中。无残留.part或解压暂存目录，试用传输客户端及临时文件已清理。它们尚未接入本项目的材料筛选/改写实验，本轮没有模型请求。

## 模型语义评分

评分角色读取 `.env` 或同名环境变量中的 `LLM_JUDGE_MODEL_NAME`、`LLM_JUDGE_BASE_URL`、`LLM_JUDGE_API_KEY`；环境变量优先。支持 API 根路径或完整 `/chat/completions`、`/responses` 地址，先规范化为 API 根路径，再按所选接口拼接一次。**请求仅由 Python 发出**，端点只保存哈希，凭据不写入快照或日志；该模型只用于评分，不自动承担生成/答题角色。

```bash
# 先估算：不调用模型，不创建评分运行
.venv/bin/python -m egvqg score --source-run stage-a-smoke \
  --run-id stage-b-score-gateway --limit 1 --max-tokens 2048 --dry-run

# 在新运行中复用已有问题、核验答案和四条件作答，仅执行语义评分
.venv/bin/python -m egvqg score --source-run stage-a-smoke \
  --run-id stage-b-score-gateway --limit 1 --max-requests 12 --max-tokens 2048
.venv/bin/python -m egvqg summarize --run-id stage-b-score-gateway
```

网关连接遵循本机内网配置：仅当主机为 `aigc-api.hkust-gz.edu.cn` 时，TCP 目标固定为 `10.121.10.250`，源地址绑定 `10.123.4.32`。URL、Host 和 TLS/SNI 保留原域名，并开启证书验证。使用 Python 标准库直连，完全忽略代理环境变量（等价于 `trust_env=False`）；不会修改全局 DNS 或 socket 解析。公共 DNS/fake-IP 结果不能用来判断此内网服务是否可达。

`LLM_JUDGE_CONNECT_IP`、`LLM_JUDGE_LOCAL_ADDRESS` 可覆盖连接地址与源地址；源地址必须真实存在于本机并有相应路由，失败时不会静默切换网络。本地 vLLM 等其他主机默认不应用上述网关绑定。

`LLM_JUDGE_API` 支持 `auto`（默认）、`responses`、`chat_completions`。auto 对 `gpt-*` / `luna*` 选择 Responses，其余选择 Chat Completions；两者分别使用 `max_output_tokens` / `max_tokens` 并规范化 token 用量。Responses 不传 `temperature`，以兼容推理模型。接口类型、网络配置与参数写入运行身份和缓存键；不在失败后自动换接口或模型。无需 OpenAI SDK，也无需 shell `source .env`。

`--limit` 限制源运行的材料数，同时保留其全部方法、失败与未决项，分母不按可评分题筛选。新运行保留原问题/答案/作答 ID 及 `source_run_id`；源运行不修改。评分依据是当前问题、其已核验答案、可接受表述与 rubric；接受语义改写及等价单位，明确区分错误和未决。评分模型看不到方法名与消融条件，不重新建立金标准。自动评分仍需人工校准。

模型输出、判定理由、调用及计费不确定性分别记录在 `calls.jsonl`、`scores.jsonl`、`requests.jsonl`、`usage.json`。未知 token/费用记为 `null`，不是零。当前串行调用；超时读取 `LLM_JUDGE_TIMEOUT`（默认 60 秒，上限 300 秒），429/5xx 重试读取 `LLM_JUDGE_MAX_RETRIES`（总尝试最多 3 次）。传输失败不自动重发；`--max-requests` 是跨恢复累计的硬请求次数上限，可提高该上限继续预算中断的运行。`--max-tokens` 默认 768；服务价格未知时仅提供请求/token 上限，不虚构金额。

缓存位于 `outputs/cache/judge/`，覆盖模型、端点哈希、解码参数、评分提示、Q、答案/rubric、回答、条件和协议。已持久化响应可跨恢复复用；请求发出后、响应写盘前的中断仍可能存在未知计费，不能保证远端恰好一次。已记录的错误/未决不静默重试；修复服务后用新 `run_id` 重评，已有有效缓存仍可复用。改变问题、评分依据、提示或模型后也必须使用新运行。

本机 `outputs/stage-b-score-gateway/` 已完成真实评分接通测试：1 组合成材料、3 个预安排问题，8 次 Responses 请求全部成功，4 个正确/4 个错误判定，另 4 条因参考答案未核验保持未决；共 3,790 tokens，费用未知记 null。先以单请求预算暂停，再恢复剩余 7 次；完整重跑追加 0 次调用、0 条记录。该结果只证明评分接通，不是评分准确率校准或研究结论。上一轮未配置内网路由的失败记录仍保留在 `outputs/stage-b-score-smoke/`，未覆盖或重写。

## 数据与题目筛选口径

材料准入见 [研究方案](VQG_Proposal.md) §5.1，问题/参考答案/回答的分层判定见 §4.4。先检查实际图像、文本与图像的必要作用、共同求解目标、答案泄露和来源隔离，再出题。问题分为“符合双模态目标的优质题 / 有效但未满足目标 / 无效 / 待审”；回答独立判“正确 / 错误或不完整 / 未决”，不混成一个总分。

当前合成图仍是占位图，mock的有效性/参考答案不是人工金标准；温度例源rubric有“活性阈值误写成℃”的问题，需在后续夹具修订时纠正。此前8次网关判分只涉及30℃与40℃，不能证明能区分问题质量。出题前按材料统一筛选，出题后所有尝试与失败仍保留在主指标分母。

## 真实材料质量诊断

当前执行指标为研究方案§4.5的 **有效性、参考答案核验、文本必要性、图像必要性、推理整合性、教育适配**。全部符合才标为自动高档；关键项未知则待审。问题标签与回答正确性独立，未决不补0。

本轮选用ScienceQA训练样本40、137、11359：前两组是含性状对应规则的杂交方格，第三组是图注背景的食物链。hint与原图均未修改；问题、参考答案与预期标签为调用模型前由AI起草的对照，原题答案/solution不参与。预期类别和参考答案不会传给质量评估或四条件答题器。各评估角色复用同一配置模型，属于诊断而非独立金标准。

```bash
# 复用已下载ZIP，显式导入诊断材料；不会下载完整数据第二次
.venv/bin/python -m egvqg download-data --split train --limit 3 --ids 40 137 11359 --offline

# 本轮已冻结的对照位于该材料目录的quality_controls.jsonl
.venv/bin/python -m egvqg diagnose-quality \
  --materials data/processed/scienceqa/hint-image-v1/train-ids_fb91b99d8c6b6040/materials.jsonl \
  --cases data/processed/scienceqa/hint-image-v1/train-ids_fb91b99d8c6b6040/quality_controls.jsonl \
  --run-id scienceqa-quality-diagnostic --limit 3 --max-requests 120 --max-tokens 2048 --dry-run
```

去掉 `--dry-run` 执行/恢复。相同运行重跑会复用记录，改变材料、图像、对照题、提示或模型则要求新run_id；此类运行用 `diagnose-quality` 重算报告，不使用仅支持旧管线记录的 `summarize`。预算预估上限114次（不含重试），实际97次请求，96次成功、1次仅Q答题超时原样保留；已返回用量41,682 tokens，超时请求用量及总费用未知。

主要产物在 `outputs/scienceqa-quality-diagnostic/`：`review.html` 可直接打开查看原图、正文、问题、证据和四条件回答；`quality_results.jsonl`、`metrics.json`、`summary.md` 为完整判定；`controls.jsonl` 与 `config.json` 固定对照和模型角色；`calls.jsonl`、`requests.jsonl`、`usage.json` 保留审计和用量。HTML默认折叠自动结论与AI预期，便于先独立审阅，未预填人工标签。

实际区分：只看有效性接受9题；再要求图文都必要接受3题；检查是否为同一推理后保留2题，排除了“问国家+问图中生物”的拼接题。自动类别与事前AI标签12/12一致，6个回答边界对照全部一致；这些不是人工准确率。原BDR为0个确认通过、3个未决、9个不通过：两道高档题与拼接题都因单模态拒答而未决，故BDR不能单独为这批题排序。48次条件答题含23次作答、24次拒答、1次技术失败，全部保留。

## 随机30组材料：能产出多少

从3,944组训练候选简单随机、不放回抽30组，seed固定为20261111；不按潜在质量挑选、不根据输出换样本。样本包含29组自然科学、1组语言科学，没有抽到社会科学，因此小样本的类别覆盖有限。图像在本批内不重复，但部分与此前材料共享图片，不能当独立校准集。

```bash
.venv/bin/python -m egvqg screen-sample --run-id scienceqa-screen-30 --limit 30 --seed 20261111 --dry-run
# 去掉--dry-run执行两轮筛查；同run_id恢复已有记录
# 完整AI逐图复核记录已保存，可一起重算，原模型判断仍单独保留
.venv/bin/python -m egvqg screen-sample --run-id scienceqa-screen-30 \
  --reviews outputs/scienceqa-screen-30/agent_reviews.jsonl
```

模型初筛/质疑审核共60次请求，原结果为15通过、6未通过、9待审。逐图AI复核发现“普通生物知识被当作必要文字”等误判，暂定为 **5通过、18未通过、7待审**。复核同样不是人工金标准，不能把模型两次同意或Codex复核称为已确认有效。

按5/30折算：训练候选约657组，近似95% Wilson抽样区间折算约289–1,324组；若7个待审全部通过，点估计约1,578组。待审边界与抽样区间是不同概念，且二者都不涵盖AI误判和去重损失。假设其他分区有相同保留率，test约210组、全量约1,089组；这只是情景外推，尚不能保证获得300组正式测试材料。

产物：`outputs/scienceqa-screen-30/summary.md`、`review.html`、`sampling.json`、`screens.jsonl`、`audits.jsonl`、`results.jsonl`、`estimate.json`、`agent_reviews.jsonl`、`reviewed_estimate.json`。所有30组和两套判断完整保留。60次请求全部完成，共103,287 tokens，费用未返回；复算新增请求/记录为0。通过项集中于符号映射+读数、性状映射+杂交格、冷却时长+数据表；五组磁铁等争议样本仍待审。

## Checklist 有界改Q

当前把检查表用于实际改写：**原题诊断 → 将缺口交给改写器 → 一次改Q → 隐去版本身份后检查 → 单独核验答案 → 保留改前/改后/未修复项**。不改原始T/I、不加入新规则、不把标准答案交给改写器。检查项目包括有效性、材料必要性、同一求解目标、题干泄露、数值依赖和单位定义；有效性不等于双模态适配。

```bash
.venv/bin/python -m egvqg refine-questions \
  --materials data/processed/scienceqa/hint-image-v1/train-n30-seed20261111/materials.jsonl \
  --cases data/processed/scienceqa/hint-image-v1/train-n30-seed20261111/checklist_reviewed_cases.jsonl \
  --run-id checklist-repair-reviewed --limit 3 --max-requests 12 --max-tokens 2048
```

已做3组真实材料：电池题改问平均每分钟降温速度，必须从T读取5分钟、从I读取温差；粒子题经Codex明确为每毫升**溶剂**对应的模型溶质粒子数，避免把溶剂体积当成溶液体积；长颈鹿体温题可由图像与常识回答，本轮未接受合规改写。2道修订题通过自动检查与单独答案核验，1道未修复保留在3次尝试分母，全部待人工核验。

过程不是一次自动成功：`checklist-repair-pilot`保留了只改措辞、且评价器混淆有效性与适配性的失败；`checklist-repair-feedback`保留反馈修订、粒子单位歧义和长颈鹿错误自评；`checklist-repair-reviewed`记录最终编辑来源与核验。`repairs.jsonl`逐项区分网关模型改写、Codex编辑和未接受的改写，不能将人工式AI编辑冒充纯模型自动收益。每个协议运行每题最多一次模型改写，不满意不会在同run_id静默重试。

最终对照见 `outputs/checklist-repair-reviewed/review.html` 和 `summary.md`；`checks.jsonl`、`references.jsonl`、`audit.json`（前两轮）保留判定与已知问题。最终核验复用了8条缓存，只发出2次新请求（1,883 tokens）；此前两轮分别11次请求/12,286 tokens和13次/14,051 tokens。尚未做独立模型/人工校准或等预算正式比较，不能据3个选定案例声称总体质量提升率。

## 可迁移的改写 pipeline 与当前指标

统一入口为 `refine-pipeline`，核心流程不读取ScienceQA特有字段。已有Q可通过 `--questions` 提供；没有Q则按配置生成一道。原题通过则保留；明确未通过或两位检查者指出同一受限缺陷代码时，最多改写一次；纯技术失败/无明确修复依据的分歧留审。改写后使用相同关卡复核，所有输入、候选、未决和失败保留。

```bash
.venv/bin/python -m egvqg refine-pipeline \
  --materials data/processed/scienceqa/hint-image-v1/train-n30-seed20261111/materials.jsonl \
  --run-id portable-refine-30 --limit 30 --max-requests 180 --max-tokens 2048 --dry-run
# 去掉--dry-run执行/恢复。离线复算不需要凭据，不调用模型：
.venv/bin/python -m egvqg refine-summary --run-id portable-refine-30
```

`configs/refinement.json` 固定生成方式与改写预算，显式配置 generator/checker/auditor/rewriter 四个角色。每个角色支持 `env_prefix` 和可选的 `model` 覆盖；对应环境变量为 `<PREFIX>_MODEL_NAME/BASE_URL/API_KEY/API/TIMEOUT/MAX_RETRIES/CONNECT_IP/LOCAL_ADDRESS`。默认四角色显式复用现有 `LLM_JUDGE` 服务，运行快照与调用日志记录身份；这不是独立模型评估。`--dry-run`不初始化provider、不要求凭据，也不会创建运行目录。配置不接受凭据字段或未实现的角色参数。

反馈只传布尔检查项、受限缺陷代码和固定修复建议，不传候选答案、rubric或评价器自由推理。代码包括尺度在比例中消去、时长未参与计算、量/单位不匹配、常识捷径、信息已写入Q及独立子问题拼接。诊断标签存在冲突时不冒充确定正确；只有明确的共同修复点才会触发一次尝试。

分母在运行前固定：N为计划材料数（每组一道原题），A为触发改写的材料数，C为成功解析且文本实际改变的题数，S为最终通过自动关卡的题数，R为改写后通过且文本确实改变的题数。实际改变按Unicode/空白规范化比较，保留科学符号的大小写；模型自报`revised`不能直接计数。

| 本轮30组开发批次的指标 | 数量 | 比例 |
|---|---:|---:|
| 改前自动存活 | 7/30 | 23.3% |
| 最终自动存活 S/N | 14/30 | 46.7% |
| 改写触发 A/N | 20/30 | 66.7% |
| 实际改写 C/N | 15/30 | 50.0% |
| 改写成功 R/A | 7/20 | 35.0% |

最终14通过、6未决（1次超时、4次输出未完成、1项有效性分歧）、7未通过、3未给出合规候选；未决不当作答错，3个无候选尝试不从分母删除。触发项中15个真实改变、2个原样返回、3个无法修复。7个新增自动通过包括6个原关卡拒绝项和1个原未决项。这是自动流程统计，不是人工存活率；已有长颈鹿题仍可被普通知识替代的抽查疑点，见 `qualitative_audit.json`，不能将46.7%当作真实数据合格率或与此前5/30的材料AI复核率直接比较。

本轮141条调用审计记录中有10次缓存复用；实际131次HTTP尝试，130次完成，1次超时保留。已返回209,077 tokens，未返回用量的请求和总费用仍未知。两次早期小额预算预检单独保留，不混入此30组计数。

产物在 `outputs/portable-refine-30/`：`plan.jsonl` 固定分母，`initial_questions.jsonl`、`checks.jsonl`、`audits.jsonl`、`rewrites.jsonl`、`results.jsonl` 保留所有阶段；`candidates.jsonl` 导出全部原题/改题及失败，`survivors.jsonl` 只导出自动通过的Q和来源键，答案核验留在审计记录中；`metrics.json`、`usage.json`、`summary.md`、`review.html` 用于查看。导出包含run_id、protocol_version、material_id、source_group_id、split和question_id，跨运行合并时按run_id与material_id关联原始材料。

### 迁移另一数据集

只需适配到现有材料契约：`material_id`、`source_group_id`、`split`、`text`、`image_path`、`image_sha256`、`provenance`、`text_source`。图片引用原文件，不复制；PNG/JPEG可直接送入当前视觉adapter。对于扁平JSONL，可用字段映射命令（以下路径是新数据集示例）：

```bash
.venv/bin/python -m egvqg normalize-materials \
  --input data/raw/other/items.jsonl --output data/processed/other/materials.jsonl \
  --dataset other --id-field sample_id --text-field context --image-field image_file \
  --split-field split --image-root data/raw/other
```

只复制明确映射的T与图像引用，其他答案/solution字段不会自动带入；文本来源是否适合研究仍须审查。缺字段、重复ID或缺失图片会在写出前报错，不静默删行。已有原题时，在同一映射命令中追加 `--question-field question_text --questions-output data/processed/other/questions.jsonl`，再把导出的文件传给pipeline的 `--questions`。也可自行提供 `{"material_id":"...","question":"...","source":"..."}` JSONL；缺少Q的材料行不会丢弃，后续按计划生成。不会自动沿用原数据集答案作为新Q的评分依据。其他文件格式先转换为这份扁平契约即可，核心生成/检查/改写代码不需要改。

`--limit`按输入顺序限制计划材料数，不负责随机抽样；需要总体估计时应先冻结独立抽样清单。旧run_id只能恢复相同材料、Q、提示、模型和协议；增加预算可继续同run_id，改变实验配置需新run_id。尚未在第二个真实数据集实测迁移，只用不同字段名的离线夹具验证了输入映射和完整流程。

## 当前未实现

三种生成方法的正式公平比较、人工适用性审查及来源/模板去重、人工标签回填、独立评分校准、正式规模实验与统计区间。当前已接入真实普通/强提示Q生成与checklist改写，但不等于完成三方法主实验。真实图像评估/四条件答题已在上述诊断中跑通；Chat Completions图像分支通过离线检查，本轮真实请求使用Responses。原合成占位图与阈值单位问题仍只存在于旧夹具，本轮未使用。
