# Evidence-Grounded Visual Question Generation

教育图文问题生成：输入一段文本与配套图像，生成一道需要联合使用材料信息的问题。核心问题是：**模型生成的问题，究竟有多少真正要求同时使用文本和图像才能回答。**

## 当前状态

- **Qwen3-VL-8B baseline（当前被评对象）**：本地卡3跑完全1257组，级联评估加必要性评估，已核实双必要下限19题（1.5%），API词元约473万；见`outputs/qwen3vl8b-necessity-v2/probe_report.md`。
- **baseline**：全1257组ScienceQA（hint + 图像）用gpt-6-luna、plain-v1生成问题并完成图文回答，评估已完成（`outputs/luna-api-baseline-test1257-*`）。原评估器自动暂定64题双必要（5.1%）。
- **必要性评估器**：`necessity-ensemble-*`（两个模型组加仲裁，答案槽位判等价）在全1257组上自动暂定38题（3.0%）。人工抽查：38题中确认双必要19、偶然不变性7、误放行12，另发现漏判4；**已核实双必要下限约23/1257（1.8%）**。误差来源、口径和限制见`outputs/necessity-v3-full-v1/probe_report.md`第八、九节。
- **报告口径只有两类**：已核实双必要与其余。“其余”不等于都是单模态问题：它包含单模态即可作答的题、问题本身不通过的题和尚未判出的题，报告里分别列出。
- **尚未做**：所提生成方法的公平比较、非教育数据（WebQA、MultiModalQA）上的泛化检验、整体重复运行评估稳定性。

## 文档入口

| 文档 | 作用 | 状态 |
|---|---|---|
| [VQG_Proposal.md](VQG_Proposal.md) | 研究定义：任务、指标、评估与实验边界 | 研究定义以此为准 |
| [story.md](story.md) | 论文叙事：研究问题、章节骨架、拟贡献 | 随研究进展修改 |
| [docs/necessity_formal_argument.md](docs/necessity_formal_argument.md) | **数学参考**：双模态必要性的定义、命题与证明；第9节为评估实现约定 | 当前 |
| [docs/references.md](docs/references.md) | **论文与数据参考**索引，附数学命题索引 | 当前 |
| [docs/evaluation_protocol_v1.0.md](docs/evaluation_protocol_v1.0.md) | 早期评估协议（五维判据、固定分母） | 历史 |
| [docs/evaluation_design_v2_draft.md](docs/evaluation_design_v2_draft.md) | 早期评估设计草案（英文） | 历史，已被数学参考与评估器取代 |
| [docs/implementation_plan.md](docs/implementation_plan.md) | 实施路线与阶段验收 | 当前 |
| [PROGRESS.md](PROGRESS.md) | 唯一进度日志：当前状态、里程碑、工作记录 | 追加式 |
| [AGENTS.md](AGENTS.md) | 供编码助手遵循的项目约定 | 当前 |

## 环境与离线检查

项目使用一个 `.venv/` 和 `uv.lock`。默认不调用真实 API、不下载数据、不训练模型。

```bash
uv sync
.venv/bin/python -m pytest          # 离线测试，全部使用mock
```

若本机 `uv` 默认缓存目录不可写，可先指定可写缓存，例如 `UV_CACHE_DIR=/tmp/uv-cache uv sync`。

### 模型与网关配置

模型调用读取 `.env` 或同名环境变量（环境变量优先，`.env` 不入Git）：

| 变量 | 含义 |
|---|---|
| `LLM_JUDGE_MODEL_NAME` / `LLM_JUDGE_BASE_URL` / `LLM_JUDGE_API_KEY` | 默认模型、API根路径（或完整`/chat/completions`、`/responses`地址）、密钥 |
| `LLM_JUDGE_API` | `auto`（默认）、`responses`、`chat_completions`；auto对`gpt-*`/`luna*`选Responses，其余选Chat Completions |
| `LLM_JUDGE_CONNECT_IP` / `LLM_JUDGE_LOCAL_ADDRESS` | 可选：固定TCP目标地址和源地址（内网网关需要）；URL、Host与TLS/SNI仍用原域名并校验证书，失败时不静默切换网络 |
| `LLM_JUDGE_TIMEOUT` / `LLM_JUDGE_MAX_RETRIES` | 超时秒数与技术重试次数 |
| `LLM_JUDGE_CONCURRENCY` | 并发上限（部分入口使用） |

网关上的模型名区分大小写与连字符（例如`DeepSeek-V4.1-Flash`）；`/v1/models`可列出实际可用名称。模型输出、判定理由、调用及计费不确定性分别记录在`calls.jsonl`、`requests.jsonl`、`usage.json`，未知token或费用记为`null`而不是0。响应缓存在`outputs/cache/judge/`。

## 工作流与命令

命令形如 `.venv/bin/python -m egvqg <命令> --run-id <run> ...`，`--run-id`、`--project-root`、`--output-root`写在子命令之后，具体参数用`<命令> -h`查看。同一`run_id`同配置再次运行即恢复，只补缺失记录；改变配置或提示须换新`run_id`。

| 阶段 | 命令 | 说明 |
|---|---|---|
| 数据 | `download-data`、`fetch-dataset`、`normalize-materials` | 见下节 |
| baseline生成 | `simple-baseline-init/run/report` | 只见hint与原图，生成问题并做独立完整图文回答；异步并发，图片有大小硬限。`run`加`--local-model-path <Qwen3-VL目录> --device cuda:0 --max-batch 16`可改用本地模型（单卡动态批处理，用`CUDA_VISIBLE_DEVICES`指定卡） |
| 基础评估 | `simple-baseline-eval-init/run/report` | 全批生成完成后才能创建；分别评可答性/忠实性、原回答支持、双必要性 |
| 基础评估（级联） | `simple-baseline-eval-init --cascade` | 有效性 → 单模态筛查 → 只对幸存题生成反例；比完整流程便宜约一半 |
| **必要性评估（推荐）** | `necessity-ensemble-init/run/report` | 见下节 |
| 方法实验框架 | `experiment-init/run/review`、`records-metrics`、`records-patterns`、`stage-two`、`reference-review`、`review-packet`、`freeze-evaluation`、`probe-*`、`retry-evaluation` | 四种生成方法（plain、strong、anchored、no_image_anchor）与四条件作答的固定名额实验，指标分母为预安排名额N；详见[实施路线第6节](docs/implementation_plan.md) |
| 诊断 | `faithfulness-review`、`baseline-analysis-init/run/report` | 对已有baseline的定向复核与忠实性派生 |
| 评分 | `score` | 对已保存的回答做模型语义评分，源运行不修改 |
| 阶段A | `run`、`smoke`、`summarize` | 用`tests/fixtures/stage_a/`合成材料的离线闭环 |

## 信息必要性评估入口（推荐）

两个模型组（默认`gpt-6-luna`与`DeepSeek-V4.1-Flash`）各自做单模态充分性筛查和反例复算，仲裁模型（默认`gpt-5.6-terra`）只处理两组没拿定的题，所有判定由代码按固定规则给出；答案等价由仅从题干生成的答案槽位决定。定义与约定见[数学参考第9节](docs/necessity_formal_argument.md)。需先有完成的`simple-baseline-eval`运行作为来源。

```bash
R=necessity-v3-full-v1; E=".venv/bin/python -m egvqg"
$E necessity-ensemble-init   --run-id $R --source-run luna-api-baseline-test1257-eval-v1   # 冻结题目、提示、模型与上限；同配置幂等
$E necessity-ensemble-run    --run-id $R --limit 10     # 先小批量真实预检；去掉--limit全量，--phase screen|arbitrate|all
$E necessity-ensemble-report --run-id $R                # 固定分母报告；离线，写results.json与final_items.json
```

- 参考答案默认由两个评估组用完整材料独立给出（一致才作参考）；被评模型较弱时不要用它自己的回答作参考（`init --baseline-reference`才会这样做）。
- 并发默认32，每次调用至多重试一次，HTTP硬上限默认16000；失败的调用留在`results.jsonl`但不缓存，恢复时补跑。
- 报告只分“确认双必要”与“未确认”；确认项必须人工复核后才计入，评估器自身会误放行（对称方格、被预设绑定的交换题等）。
- 只有原评估器给出两个方向假想情形的题才可能被确认双必要，所以默认只评估这些题（约占可答题的7%，却是确认项的全部来源）；`init --all-items`才会评估其余题，仅作诊断，花费约为10倍。

## 数据

```bash
.venv/bin/python -m egvqg download-data --split train --limit 3          # ScienceQA官方来源；--offline可离线重新导入
.venv/bin/python -m egvqg fetch-dataset --dataset multimodalqa --workers 2
.venv/bin/python -m egvqg fetch-dataset --dataset webqa --workers 8 --proxy http://127.0.0.1:7890 --image-mirror
```

- **ScienceQA**：[官方下载源](https://github.com/lupantech/ScienceQA#-download-the-dataset)，元数据固定到提交`2cbf8318e07b9ece895bb2ae605e71e38d623264`；默认只用非空`hint`与原图，原题、选项、答案、`lecture`、`solution`不进入材料。原始数据在`data/raw/scienceqa/<revision>/`，产物在`data/processed/scienceqa/`。非空图文只是规则初筛，不能据此认定具备双必要性。
- **MultiModalQA / WebQA**：[AllenAI MultiModalQA](https://github.com/allenai/multimodalqa)、[WebQnA WebQA](https://github.com/WebQnA/WebQA)。`fetch-dataset`只下载、解压、校验，来源URL、大小、SHA256和记录数写入各目录的`manifest.json`；`.part`可恢复。两者已完整下载（MultiModalQA约4.6GiB，WebQA约119GiB），尚未接入材料筛选和评估。
- 其他数据集的平铺JSONL用`normalize-materials`映射到共享的T/I材料契约。

## 提示与配置

`prompts/*.txt`是独立维护的提示模板，`$变量`占位符缺失即报错，每个run把用到的模板快照进`outputs/<run>/prompts/`。`configs/`放不含凭据的运行配置：`stage_a.json`（阶段A）、`experiments.json`与`experiments_review_gated.json`（方法实验）、`necessity_assessment_v2.json`（基础评估使用的知识范围K与模板）。

## 当前未实现

所提生成方法的正式公平比较、独立人类标注与评分校准、非教育数据上的泛化检验、评估器的整体重复稳定性运行。已接入真实普通/强提示Q生成，但不等于完成方法主实验。
