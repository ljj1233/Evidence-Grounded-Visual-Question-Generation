# 项目进度

本文是唯一的进度入口：当前状态、里程碑、按时间的工作记录。2026-11-11至11-12的旧记录已压缩，保留日期、数字、裁定、失败与产物；压缩前的完整原文在Git提交`3aff4eb`的`PROGRESS.md`中。

## 当前状态（2026-11-13）

- **当前被评对象是本地Qwen3-VL-8B的全1257组baseline**（luna只是调试参照）：已核实双必要下限19题（1.5%），评估与审计见`outputs/qwen3vl8b-necessity-v2/probe_report.md`；API词元约473万，比luna那批便宜约66%。
- **研究主线**：模型拿到文本和图像后生成的问题，有多少真正要求同时使用两种信息才能回答。ScienceQA（官方test、hint + 图像共1257组）是当前载体；当前阶段是baseline现状分析，不是所提方法的比较。
- **baseline**：gpt-6-luna、plain-v1只见hint与原图生成问题，并独立完成完整图文回答；全1257组处理完成（生成1256题、1237作答、18拒答、1生成失败、1回答失败），目录`outputs/luna-api-baseline-test1257-simple-v1/`。
- **原评估器**（`simple-baseline-eval-*`，`outputs/luna-api-baseline-test1257-eval-v1/`）自动暂定：可答1100、不通过118、可答性待裁定39；1100道可答题中64图文都必需、976至少一种可省略、60待裁定。固定1257下64通过（5.1%）。
- **新必要性评估器**（`necessity-ensemble-*`，`outputs/necessity-v3-full-v1/`）：两个模型组加仲裁，答案槽位判等价。自动暂定38通过（3.0%）、1053不通过、166待裁定。人工抽查（claude-code，非盲，user_delegated_ai）：38题中确认双必要19、偶然不变性7、误放行12，另发现漏判4；**已核实双必要下限约23/1257（1.8%）**，偶然不变性类另约10题。
- **报告口径只有两类**：已核实双必要 / 其余。“其余”包含单模态即可作答（935，其中只用图像412、只用文本385、各自都够138）、问题本身不通过（118）和未判出（待裁定166、无问题或回答1）。
- **已知系统性误差**（重复运行消除不了）：交换午餐类（文本写明想要什么，加题干预设“存在可行交换”后文本即定答案）、温度表类（题干重述了图中信息）、性质列表与图注型（答案是单一模态内容的转抄）、答案为1/2的对称方格（偶然不变性）、模型用外部知识把本应自由的对应当成已知（Q0949）。共同本质：评估器没有真正把“合法补全空间”落实，反例由LLM自由生成并由LLM判合法。
- **代码**：`src/egvqg/`共20个模块、98个离线测试通过；已移除不再被当前流程引用的旧模块（四条件之外的旧探针、改写、筛选、本地推理等）。仓库有3个提交，未配置远端。
- **数学参考**：[docs/necessity_formal_argument.md](docs/necessity_formal_argument.md)；论文与数据参考：[docs/references.md](docs/references.md)。

## 后续里程碑

1. 决定是否把评估器的反例改成变量级结构化输出（每个反例写明改变的变量与被固定的约束，由代码核对），或维持“候选生成加全量人工复核”的口径。
2. 扩大对“可省略”（935题只抽查了27）和必要性待裁定（126题只抽查了15）的复核范围，估计漏判。
3. 非教育数据（WebQA、MultiModalQA，原始数据已下载）上的泛化检验：各抽约30组，沿用同一套评估器，检查各模块有没有失效。
4. 整体重复运行一遍评估，检查归属稳定性（小试验约88%）。
5. 在同模型、同材料下比较证据提取、跨模态配对、证据锚定等构造方式，看能否提高达标题数且保持忠实性。
6. 论文：叙事见`story.md`，研究定义见`VQG_Proposal.md`。

## 工作记录

### 2026-11-11：固定工程协作约定
- 做了什么：更新 `AGENTS.md` 与 `docs/implementation_plan.md`，固定代码/数据/运行产物/缓存的定位与协作约定。
- 结果与决策：本文件采用“当前状态原位更新 + 工作记录末尾追加”；只保留关键测试，批处理优先缓存与恢复；不新增平行总结文档。
- 失败与限制：仅文档检查，未执行研究代码或真实模型调用。
- 产物：`AGENTS.md`、`docs/implementation_plan.md`

### 2026-11-11：完成阶段 A 工程骨架与离线闭环
- 做了什么：实现 `src/egvqg/` 的记录、提示、provider、三种生成、核验、四条件作答、评分、指标、管线和 CLI，并用合成夹具跑通离线闭环。
- 结果与决策：指标以预先安排的尝试数为固定分母，无效/生成失败/技术失败留在分母，未决单列给上界，未决不按答错；无材料正确率只用“有效+答案已核验+仅Q可评分”分母。`pytest` 4 passed；`smoke` 合成 N=18（有效11、无有效5、未决2；BDR 2/18、BDR未决3；无材料5/10）；同 run_id 重跑追加0条。
- 失败与限制：全部为合成工程数据，不是模型实验或方法结论；真实视觉 adapter、图像输入审计、缓存/重试/预算、ScienceQA导入筛选、人工审查导入导出、统计区间、正式实验均未实现。
- 产物：`configs/stage_a.json`、`prompts/`、`tests/fixtures/stage_a/`、`outputs/stage-a-smoke/`

### 2026-11-11：ScienceQA 下载与独立模型评分入口
- 做了什么：实现 `dataset.py` 下载 ScienceQA（固定提交 `2cbf8318e07b9ece895bb2ae605e71e38d623264`，训练图像 ZIP 约600 MiB）、`JudgeProvider` 语义评分与 `score` CLI。
- 结果与决策：候选按官方 split、非空 hint 与 image、固定种子选择，原题/选项/答案/lecture/solution 不进入材料，人工审查标记 pending；评分三值严格校验，API/解析/截断失败均未决。`pytest` 15 passed；阶段A smoke 仍 N=18、BDR=2/18。
- 失败与限制：实际评分 smoke（`stage-b-score-smoke`）8次请求均未响应（TLS EOF），4条因参考答案未核验不调用，12条评分均未决；公共 DNS 返回 NXDOMAIN，未绕过 TLS 校验。未验证语义评分准确率，未做正式实验。
- 产物：`data/raw/scienceqa/<revision>/`、`data/processed/scienceqa/hint-image-v1/train-n3-seed0/`、`outputs/stage-b-score-smoke/`

### 2026-11-11：修复内网网关连接并完成真实评分 smoke
- 做了什么：按用户补充的网关规则，`JudgeProvider` 改为标准库 HTTP(S) 直连，固定目标 `<网关IP>`、源地址 `<源IP>`，Host/SNI/证书校验用原域名，不用代理。
- 结果与决策：`LLM_JUDGE_API=auto/responses/chat_completions`，默认 gpt-/luna 用 Responses。`stage-b-score-gateway` 共8次成功请求、3,790 tokens、约9.715秒，费用未知；12条评分中4对、4错、4未决（参考未核验）；BDR=1/3、未决1，仅合成 smoke。`pytest` 20 passed；完整重跑新增0。
- 失败与限制：Chat Completions 分支只离线通过，未调用其他模型验证；未启动正式实验。
- 产物：`outputs/stage-b-score-gateway/`

### 2026-11-11：明确筛选判据并审计现有质量证据
- 做了什么：用户要求先明确材料与问题/答案高低判据；在 `VQG_Proposal.md` §4.4 与 §5.1 细化操作检查，并审计网关运行的3题证据。
- 结果与决策：保持 v3.0 指标定义、三值语义和固定分母，不加加权总分；问题分“优质题/有效但未满足双模态目标/无效/待审”；缺证据不等于低质量。
- 失败与限制：温差例是占位图，plain/strong 目标相近，四条件回答由脚本设定，8次判分只涉30℃/40℃，plain rubric 把45.75误写为45.75℃；不能据此确认质量排序或评分校准。尚未证明评分器能区分题目高低。
- 产物：`outputs/stage-b-score-gateway/quality_audit.md`

### 2026-11-11：真实ScienceQA材料上的质量指标诊断
- 做了什么：在真实 ScienceQA 材料（40、137、11359 等）上构造12道问题和6个回答对照，新增 `quality.py` 与 `diagnose-quality`，指标 V/R/D_T/D_I/S/E 按门槛分类（`quality-diagnostic-v1`）。
- 结果与决策：自动高档2、有效非目标7、无效3、未决0，与AI预期12/12一致，回答对照6/6一致；V单独接受9题，V+D_T+D_I接受3题，完整条件接受2题；c11是需要两模态但无联合推理的反例。四条件48次作答：23 answered、24 refused、1 technical_failure（c04仅Q超时）；BDR确认通过0/12、未决3/12、不通过9/12。97次请求（96完成）、41,682 tokens；`pytest` 30 passed。
- 失败与限制：标签由AI起草、各角色同模型，不称人工准确率，不外推；人工金标准、盲评一致性、独立校准、三方法生成实验均未完成；全图文已评分9/9正确不能替代问题质量。
- 产物：`data/processed/scienceqa/hint-image-v1/train-ids_fb91b99d8c6b6040/`、`outputs/scienceqa-quality-diagnostic/`（含 `review.html`）

### 2026-11-11：随机30组材料估计规模
- 做了什么：从3,944训练候选简单随机不放回抽30组（seed=20261111），每组联合见证搜索+质疑审核，并逐图做非盲Codex复核。
- 结果与决策：学科自然科学29、语言科学1；60次请求、103,287 tokens；模型初判15通过/6未通过/9未决；Codex复核暂定5通过/18未通过/7待审（均 human pending）。通过率5/30=16.7%，待审全通过40.0%；训练点估计657组，待审全通过1,578组；Wilson区间7.3%–33.6%（289–1,324组），合并范围约289–2,275组；test约210/503、全量约1,089/2,613为同保留率情景。
- 失败与限制：8762与5285共享原图，非完全未见；两套标签都非人工真值，18项与模型原输出不同不等于18个确认错误；300正式测试材料目前不能保证；数字只指导预算，不含AI误判和去重损失。
- 产物：`data/processed/scienceqa/hint-image-v1/train-n30-seed20261111/`、`outputs/scienceqa-screen-30/`（`estimate.json`、`reviewed_estimate.json`、`review.html`）

### 2026-11-11：按checklist实际改Q并保留失败证据
- 做了什么：按用户补充方向新增 `refine-questions`，流程为原题检查→缺口反馈→一次Q-only改写→隐去版本身份再检查→单独参考答案核验。
- 结果与决策：v1 暴露仅删措辞不改变依赖；v2 将缺口传给改写器；最终保留电池平均速率题（答案2.6与2.0°C/min）和Codex明确的粒子题（0.20与0.10）；长颈鹿题记 no_proposal。`pytest` 40 passed。
- 失败与限制：两题仍 human pending；不能把模型对原长颈鹿题的高档误判当真值，不能据3个选择案例宣称总体改善率；checklist 不能仅凭同模型自评勾选宣布通过。
- 产物：`outputs/checklist-repair-pilot`、`checklist-repair-feedback`、`checklist-repair-reviewed`（`review.html`）

### 2026-11-11：可迁移pipeline与30组存活/改写统计
- 做了什么：新增 `src/egvqg/refinement.py`、`configs/refinement.json`、`audit_question.txt` 与通用 `normalize-materials`，并在固定30组上跑 `portable-refine-30`。
- 结果与决策：初始自动通过7/30（23.3%），最终14/30（46.7%）；20触发、15实际改变、2原样返回、3无法修复；7个改写后通过（触发项35.0%）；最终14通过/6未决/7未通过/3无候选，N=30。实际131次HTTP、141条调用审计含10次缓存复用，已知209,077 tokens。`pytest` 51 passed。
- 失败与限制：未决原因为1次checker超时、4次输出未完成、1项有效性分歧；scienceqa_1699 改写后仍可能凭常识回答（非盲AI抽查）；46.7%不能外推为真实可用数据量，也不是BDR或人工合格率；预检 `portable-refine-scienceqa30`、`portable-refine-checklist30` 未完成，不混入统计；人工存活率未知。
- 产物：`outputs/portable-refine-30/`（含 `survivors.jsonl`、`review.html`）、`configs/refinement.json`

### 2026-11-11：开始下载MultiModalQA与WebQA原始发布
- 做了什么：固定官方提交（MultiModalQA `4dd14328c6d02a4daa357cc6032915a0b14602e3`，WebQA `81cc377aaddb2cf9de98cf082a2795ce53ccc1f4`）并新增 `downloads.py` 与 `fetch-dataset`。
- 结果与决策：MultiModalQA 完成（图像ZIP 2,362,869,144字节，57,058条图像路径均存在；问题 train23,817/dev2,441/test3,660）；WebQA 标注包与索引完成（train+val 41,732、test 7,540）；官网旧CMU源不可用，改用 `Ancci/webqa_large` 镜像提交 `5d2222ebafe1443d2e31f396a5eafb8803394146` 逐卷校验。
- 失败与限制：WebQA 图像下载尚在进行，完成还需分卷全通过、7z解压、`imgs.tsv` 索引及引用核验；本轮无模型请求。
- 产物：`data/raw/multimodalqa/<revision>/`、`data/raw/webqa/<revision>/`（`source_files.json`、`image_mirror.json`、`download_status.json`）

### 2026-11-11：完成两个原始数据集下载、解压及完整引用核验
- 做了什么：补齐 WebQA 尾卷048（4路Range小段）、解压并核验全部图像引用，复验 MultiModalQA。
- 结果与决策：WebQA 51卷共54,445,524,319字节全部通过SHA256；`imgs.tsv` 72,234,224,104字节、389,750条记录、SHA256 `4c4e79a588ba198cf6fc04d3dac20c0f1d8911bba2f8c3ca34ace2a5fbeafcca`；train+val 350,777、test 90,760个唯一图像ID均可索引；问题 train36,766/val4,966/test7,540共49,272。MultiModalQA 题目共29,918。`pytest` 56 passed。
- 失败与限制：尾段曾试 huggingface-hub，未作最终来源，已从依赖移除；下载完成不等于已按严格协议筛成有效样本。
- 产物：`data/raw/multimodalqa/4dd14328c6d02a4daa357cc6032915a0b14602e3/manifest.json`、`data/raw/webqa/81cc377aaddb2cf9de98cf082a2795ce53ccc1f4/manifest.json`

### 2026-11-11：研究方案全文叙事与学术表达修订
- 做了什么：原位改写 `VQG_Proposal.md`（保留9个主节与编号小节）并同步 `story.md`，末尾附5项 Key Revisions。
- 结果与决策：协议仍为 v3.0，BDR 公式、固定N、拒答未决、四条件隔离、答案绑定、30/300组设计及核心消融不变；限制集中于§7；SynMQG 仅按公开题名定位，不补写未查证细节。
- 失败与限制：仅文档变更，未调用模型。
- 产物：`VQG_Proposal.md`、`story.md`

### 2026-11-11：按Rethink定位重整论文逻辑与骨架
- 做了什么：按用户指令更新 `story.md`，形成诊断命题与方法命题、四项交付、八节结构。
- 结果与决策：区分代表性方法的现状诊断与同基座/材料/总预算的构造原则比较；开发批次数字不写成正式结论；“首次”限定为全流程整合层面的拟投稿主张。
- 失败与限制：未改现行协议，无新增测试或实验。
- 产物：`story.md`

### 2026-11-11：升级为ICML捷径诊断与评测重审叙事
- 做了什么：原位更新 `story.md`，开篇改为多模态捷径与评测目标错位，用反事实输入干预解释四条件。
- 结果与决策：区分固定答题器的行为依赖与人工审核的信息必要性；统一“构造性基线”；不增加结构因果模型、内部机制识别、训练或额外实验。
- 失败与限制：实证发现与缓解价值按正式结果待填，不把开发统计写成结论；未把QG联合表示目标等同于作答必要性。
- 产物：`story.md`

### 2026-11-11：收束BDR术语、诊断范围与出题侧定位
- 做了什么：按用户要求将“因果代理”统一改为“反事实行为代理”，限定贡献范围，补入VQA v2与 VideoQA 视觉依赖审计文献（arXiv:2606.30220）。
- 结果与决策：删除未限定的“普遍”设问；贡献第一条限定 ScienceQA 教育图文与所测代表性范式；BDR 保持模型行为代理属性，不识别内部因果机制。
- 失败与限制：训练信号影响及迁移价值保持条件性研究启示；未改实验、协议、代码。
- 产物：`story.md`

### 2026-11-11：梳理指标与生成模型的最小诊断实验
- 做了什么：用户询问下一步实验，核对运行报告、提示与生成入口后在实施方案阶段C补充执行建议。
- 结果与决策：建议先复用12题及四条件原始回答核验必要性/整合性并独立评价相关性，再1–3组真实生成冒烟，再固定约30组合格材料比较普通与强提示；checklist 改写统计不替代三阶段效果。
- 失败与限制：相关性评分与必要性直接比较、普通/强提示公平诊断尚未完成；BDR低值不能独立区分模型/材料/评价问题；已有随机30组不能假定全有联合出题空间；未调用模型。
- 产物：`docs/implementation_plan.md`（阶段C补充）

### 2026-11-11：新版工程底座、dev冒烟、12题自动诊断与60组准入准备
- 做了什么：新增 `experiments.py` 与 `configs/experiments.json`（15个 `*-v1.txt` 模板）、锁/原子更新记录及新CLI（`experiment-init/run/review`、`records-metrics`、`probe-reproducibility`、`freeze-evaluation`、`prepare/freeze-admission`）。
- 结果与决策：`outputs/prompt-dev-smoke/` 15次请求（含3次一致性测试）；`blindspot-12-v1` 130次请求、101,286 tokens、1条格式失败；`scienceqa-admission-60-v1` 60候选、零模型调用。`pytest` 70 passed。
- 失败与限制：网关未确认严格贪心，三次一致仅为观测；12题无人工金标准、评价口径未冻结；60组无人审，不能宣称已有30组合格材料或120名额；未启动阶段二/三生成。
- 产物：`outputs/prompt-dev-smoke/`、`outputs/blindspot-12-v1/`、`outputs/scienceqa-admission-60-v1/`

### 2026-11-11：用户委托复核、协议与材料冻结，启动阶段二
- 做了什么：用户明确授权 Codex 承担审核，复核12题、扩展准入至30组并冻结协议与120名额。
- 结果与决策：12题有效9/12，必要性明确失败6/12，另2个联合题与1个拼接题；累计132审阅后30通过/15待审/87排除；3×5重复性检查15次请求/4,635 tokens；协议冻结于 `docs/evaluation_protocol_v1.0.md`；金标准范围为用户委托AI复核，非独立人类标注。`pytest` 72 passed。
- 失败与限制：不虚构 temperature/do_sample 生效；会话历史暴露如实记录；正式阶段二已启动但结果待追加，未声称方法效果。
- 产物：`docs/evaluation_protocol_v1.0.md`、`delegated_review.jsonl`、`disagreements.md`、`summary.md`、`cases.md`

### 2026-11-12：继续阶段二并核对冻结完整性
- 做了什么：用户要求 continue，确认后台进程正常运行（plain 已完成26/30，strong 未启动），验证冻结契约与首60条前缀 SHA256 一致。
- 结果与决策：未重复启动或请求；固定的16个后续抽查名额未改动。
- 失败与限制：阶段二尚未完成，未写成已完成。
- 产物：`outputs/scienceqa-admission-60-v1-plan/integrity.json`

### 2026-11-12：阶段二60题完成、事前16题复核与验收
- 做了什么：plain30→strong30 批处理完成，并按事前固定每方法8题逐图复核。
- 结果与决策：自动有效27/30与24/30，BDR 1/30与0/30（strong有效性另2未决、BDR另4未决）；717条逻辑调用（709成功、8解析失败）、678,755 tokens、2,842.386秒；16题13有效/3未决，10题存在至少一项标签差异（source=user_delegated_ai）；地图Massachusetts被误读为New Hampshire、同为10球的两瓶被误数成10/9、拼接审核把联合交换当拼接。
- 失败与限制：8个失败落在2个strong名额（JSON转义与截断），未重生成；抽查N=8/方法不冒充30题人类结论；完整锚定/消融未运行，自动差值不支持直接宣称必要性增益。
- 产物：`outputs/scienceqa-admission-60-v1-plan/{stage_two_summary.md,stage_two_metrics.json,stage_two_cases.md,acceptance.json}`、`calibration/`

### 2026-11-12：接手诊断、参考答案审核门控与分阶段执行
- 做了什么：核对阶段二记录，亲自确认 scienceqa_13534（两瓶均10球）与 scienceqa_14856（Massachusetts）两处参考错误，新增 `configs/experiments_review_gated.json`（`reference_policy=review_gated`）、`reference-review`、`experiment-run --stages`、`json_repair=escape_invalid_backslashes`、`records-patterns`。
- 结果与决策：8次解析失败中7次是LaTeX反斜杠未转义，1次是2048 token截断；瓶颈是参考答案与审核有效性而非吞吐。`pytest` 74 passed；冒烟 `outputs/gated-dev-smoke/` scienceqa_40×anchored，10次请求、6,772 tokens，停在 `awaiting_reference_review`；已初始化未执行 `outputs/anchor-headroom-dev4-v1/`（scienceqa_19、192、1992、15694×strong/anchored共8名额，诊断集非代表性样本）。
- 失败与限制：未决——BDR对拒答的处理使锚定改进可能只表现为未决增多，是否另立协议版本由用户决定；正式30组中20组属两类模板；其余44道阶段二题自动参考未审核；网关回放仅由耗时推断，未向网关方确认；源码未纳入Git，改动无版本记录。
- 产物：`configs/experiments_review_gated.json`、`outputs/gated-dev-smoke/`、`outputs/anchor-headroom-dev4-v1/`

### 2026-11-12：执行8名额dev探针的生成与参考阶段，交接仅来源复核
- 做了什么：用户指示在 `anchor-headroom-dev4-v1` 只运行生成与候选参考，并新增 `review-packet` 与 `experiment-run --attempt-ids`。
- 结果与决策：28次HTTP（上限130）、31,500 tokens、148.3秒；8名额全部产出问题；候选参考6 answerable，气体材料 scienceqa_19 的 strong 与 anchored 均被标 ambiguous（平均速率不等于均方根速率）。`pytest` 76 passed。按用户澄清更正：阶段二样例可确立捷径存在，不可靠的是比例与方法比较；锚定失败不说明材料无联合潜力；BDR定义与拒答未决语义不变。
- 失败与限制：候选参考未经审核不作结论；Claude未填任何标签；等待 Codex 先封存 T/I/Q 判断再裁定参考。
- 产物：`outputs/anchor-headroom-dev4-v1/source_review/`（`packet.html`、`labels_template.jsonl`、`key.jsonl`、`README.md`）

### 2026-11-12：Codex封存8题来源复核并裁定参考
- 做了什么：Codex 只读匿名T/I/Q和4张原图，封存标签后才打开方法映射，并裁定候选参考。
- 结果与决策：6有效、2气体模型前提未决；参考4接受、2 corrected（S01/S06评分范围收紧）、2 unresolved；独立手工见证气体质量密度B:A=11:7；来源 user_delegated_ai。
- 失败与限制：此前见过 Claude 的两气体候选歧义汇总，暴露如实保留，不宣称完全盲或人类金标准；human 保持 pending；无行为结果或方法效果结论。
- 产物：`outputs/anchor-headroom-dev4-v1/source_review/{labels.jsonl,labels.seal.json,reference_decisions.jsonl,adjudication.jsonl,report.md,acceptance.json,witness_questions.jsonl}`

### 2026-11-12：Codex直接完成八题探针下游与失败机制诊断
- 做了什么：用户明确要求自行解决，Codex 对6个 verified 且 V=1 的名额运行 answers/scores，2气体题保留未决未作答。
- 结果与决策：新增35次请求/13,030 tokens；全run 63次请求、44,530 tokens；24条件中11作答正确、13拒答未决，评分复核一致。strong 与 anchored 各有效3/4、有效性未决1/4，BDR均0/4，BDR未决分别2/4、1/4。失败机制：遗传题 strong 需T映射与I方格；气体配对漏查速率分布、溶液相对比较抵消编码单位、电池配对把不参与幅度计算的时长当必要条件；证据提取把Ii读成ii。
- 失败与限制：诊断集不提供方法优劣结论；不把更少未决解读为改进，也不因锚定失败否定材料潜力；不扩正式60名额。
- 产物：`outputs/anchor-headroom-dev4-v1/{diagnostic_report.md,diagnostic_table.jsonl,diagnostic_metrics.json,outcome_patterns.json,score_review.jsonl,probe_acceptance.json}`

### 2026-11-12：按探针任务修复配对/题干保护并有限迭代
- 做了什么：新增 `prompt_names` 与 `experiment-init --prompt-override ROLE=VERSION`，以及 pair-v2/v3/v4、generate-anchored-v2、extract-image-v2，在 dev4 上迭代三轮。
- 结果与决策：v2 来源审核把对称50%题误判T必要，后由 `amendments.jsonl` 追加更正，旧dev对应结论撤回；v3 气体/电池联合；v4 N=4 均生成并裁定V/T/I/整合通过，16条件6回答正确/10拒答未决，BDR 0/4、未决2/4，TI 4/4、T 0/4、I 2/4、0 为 0/4。共79新请求/79,011 tokens，复用20次缓存。`pytest` 77 passed。
- 失败与限制：v3 溶液/遗传配对输出截断无Q；v4 气体I条件未经支持假定等体积、溶液I条件默认一球一粒子而猜中，不能倒推为泄露，也不改评分；顶部i仍误记I；均为已见dev材料调参，非持出验证，三轮各N=4不混成成功率。
- 产物：`outputs/anchor-dependency-dev4-v2/`、`-v3/`、`-v4/`（`v4/probe_report.md`、`acceptance.json`）

### 2026-11-12：评价方法重新设计（仅设计，未运行）
- 做了什么：按用户最新优先级阅读冻结协议与历史记录，写成 `docs/evaluation_design_v2_draft.md`。
- 结果与决策：必要性判在参考解答证据上，“不必要”须给不用该模态的完整解法或不变性论证，“必要”须给反事实见证，两者皆无则未决；四条件作答改作替代解法搜索；BDR定义与拒答未决语义不变，仅作行为诊断；每题至多11次调用；验证方案为24道手写受控题（12开发/12持出），至多264次请求。
- 失败与限制：待用户决定——图例默认约定是否算通常知识、课堂模型题的有效性、Q要求的解释是否计入完整答案、参考审核覆盖面、24题预期标签由谁审核；未调用模型、未改代码。
- 产物：`docs/evaluation_design_v2_draft.md`

### 2026-11-12：评价规范草案第2版（仅规范，未实现、未运行）
- 做了什么：按用户七点修改意见原位重写草案，必要性针对问题在全部合法解法上的性质，见证改为被移除模态的两个自洽、可答、答案不同的完整内容。
- 结果与决策：流程缩减为绑定→参考→审核→每模态一次充分性/见证分析→结构校验→隔离作答→语义审核→裁定；四条件评分与BDR降为可选次要诊断；首个探针8道受控题，计划14次调用，累计HTTP硬上限24；预期标签由Claude按构造给出，须审核封存后才可运行。
- 失败与限制：待用户确认 K3（大写字母表示显性等位基因）与 N4（图例编码单位属材料事实）取舍、被要求的解释是否计为答案要点、预期标签与证据项审核者；未调用模型。
- 产物：`docs/evaluation_design_v2_draft.md`

### 2026-11-12：修正评价逻辑并完成八题首轮验证
- 做了什么：用户授权后修正草案 revision3 并封存，新增 `necessity.py`、`configs/evaluation_v2.json`、`modality-necessity-v2.txt`、`solve-v2.txt`，在8个固定控制题上执行。
- 结果与决策：基线14调用，18原始标签与预期一致但仅9项证据结构合格；追问10请求后24硬上限生效；最终11项证据合格、7项未决。N=8：供入参考V 7通过/1未决；T必要1通过/2不通过/5未决；I必要5通过/3未决；整合0通过/1不通过/2不适用/5未决。24次HTTP、58,273 tokens、226.478秒。`pytest` 90 passed。
- 失败与限制：E3/I 见证两种答案均1而无效，E2/I 追问答反模态；整体未达扩量门槛；E7双必要但拼接，不能把整合失败当双模态不必要；E8只验证门控；AI审核一致性不称人类准确率；来源 user_delegated_ai，暴露对Claude预期的了解。
- 产物：`outputs/evaluator-necessity-probe1/{evaluation_report.md,comparison.json,results.json,acceptance.json,usage.json}`、`data/processed/scienceqa/evaluator-probe1/materials.jsonl`

### 2026-11-12：首轮必要性评估器复核（Claude，仅检查与设计）
- 做了什么：Claude 逐条读取 probe1 原始输出对照 `metrics.validate_necessity_item`，未调用模型、未改任何产物。
- 结果与决策：18项中9项结构通过，其余9项中8项为接口问题且证据核心成立（多填path 4项：E1/T、E2/I、E6/T、E6/I；`removed`照抄占位“T或I”2项：E2/T、E4/T；来源写K1而非K 2项：E5/T的c1、c2），1项真实推理失败（E3/I）；E2/I追问答成移除T，且“取最后一轮”覆盖了基线中核心成立的见证；`supplied_by` 只有question/general_knowledge两取值导致捷径误标。建议（待用户确认）：零调用派生重评 norm-v1，再收紧合同做26次调用（硬上限30）的新run，含6道新控制题。
- 失败与限制：Claude是这8题及预期标签的构造者，非独立盲审；首轮不能说明评估器准确率、自然题适用性或脱离复核者的可靠性。

### 2026-11-12：零调用规范化复评与13题合同v3验证
- 做了什么：用户授权，新增 `evidence_contract.py` 的 norm-v1 与 v3 绑定，Part1 零调用复评、Part2 冻结 `configs/evaluation_v3.json` 与 `modality-necessity-v3.txt` 跑13题。
- 结果与决策：Part1 17项可用，E3/I 与 E2/I 仍拒，4个附注path均非合法无模态解法，新增API 0。Part2 26计划调用，24次HTTP/2缓存、52,010 tokens、157.550秒；30项预列组件判断中15项结构合格、严格审核后11项确认、19未决。结论：证据法值得保留，当前v3接口不适合扩批。`pytest` 98 passed。
- 失败与限制：10条逻辑响应JSON解析失败，14/26单位原生格式失败，超过预设最多1项，未救援未重跑；E3/N6复用响应错称豚鼠、E3显隐性措辞冲突、E6误将5分钟归于Q、N3粒子数与符号数混淆记语义暂缓；N5/N6干扰方向JSON失败，双胞胎不变性未证实；审核有预期暴露，非盲审。
- 产物：`outputs/evaluator-necessity-norm1/`、`outputs/evaluator-necessity-probe2/`、`configs/evaluation_v3.json`

### 2026-11-12：整理问题存在性的第一步证据与通用故事定位
- 做了什么：对照 scienceqa_9369 原表与运行记录，原位更新 `story.md` 开头与§1.1。
- 结果与决策：scienceqa_9369 普通Q在TI/I条件均正确，T/0拒答保留未决；Q+I中两个23.7−17.7计算已覆盖完整答案；已确认的存在性与未确认的比例、泛化、自动评价可靠性、锚定效果分开。
- 失败与限制：摘要和方法仍标研究计划；本轮无模型请求、无代码或协议改动。
- 产物：`story.md`

### 2026-11-12：执行答案可辨识性的有限情形真实模型探针
- 做了什么：新增 `identifiability.py` 与 `identify-answers-v1` 模板，调用前冻结11题/22分析/26 HTTP硬上限，跑 `identifiability-dev-v1`。
- 结果与决策：22次HTTP，14,448 tokens，59.969秒，技术/解析失败0；22/22答案集合与预声明有限空间相同；模型给出电池时长改变致0.6/0.3变化及遗传映射改变仍为1/2的相反信号。`pytest` 105 passed。
- 失败与限制：8合成题+3既有题不冒称自然样本；封闭空间与答案组件由构造者给出，不证明开放域可靠性或自动语义分解。
- 产物：`outputs/identifiability-dev-v1/{probe_report.md,pre_run_plan.json,cases.jsonl,oracle.jsonl,records.jsonl,calls.jsonl,results.json,usage.json,acceptance.json}`、`data/processed/identifiability-dev-v1/`

### 2026-11-12：撤掉人工取值范围并测试八个自然措辞问题
- 做了什么：新增 `open_necessity.py` 与 `open-necessity-v1`，4组日常材料8问题，不给取值范围，模型自行给两份合法情形或唯一解。
- 结果与决策：16次HTTP，16,707 tokens，60.157秒；12份ambiguous见证、4份unique解法经Codex逐项复核均支持；参考40.8元、36元、60元、30元、48瓶、2倍、B口200米、B口。`pytest` 106 passed。
- 失败与限制：构造/审核者知预期、图清楚、目的性强，不称自然题准确率；四个联合、三个图像充分、一个文本充分是构造组成，不作发生率；下一步优先真实未设计题与负对照。
- 产物：`outputs/open-necessity-dev-v1/{questions.html,probe_report.md,pre_run_plan.json,review_key.jsonl,records.jsonl,calls.jsonl,evidence_reviews.jsonl,arithmetic_checks.json,results.json,usage.json,acceptance.json}`、`data/processed/open-necessity-dev-v1/`

### 2026-11-12：开放取值探针扩至20个新问题
- 做了什么：用户授权扩到20，新建 `open-necessity-dev20-v1`（10组新图文×2题），预算40分析/44 HTTP。
- 结果与决策：40次HTTP，44,134 tokens，165.651秒；40方向判定与封存预期一致，38证据支持；18/20题两方向证据均确认；双必要8是/11否/1未决，T必要9通过/10否/1未决，I必要17通过/2否/1未决，N=20。
- 失败与限制：Q09移除T的最大摆放数依赖未明确单排、Q10移除I的替代图无尺寸标注是否满足Q含糊，严格暂缓2项，是审核敏感边界不宣称确定独立错误；目的性构造和知预期AI复核不能当自然总体准确率。
- 产物：`outputs/open-necessity-dev20-v1/{probe_report.md,questions.html,pre_run_plan.json,review_key.jsonl,records.jsonl,calls.jsonl,evidence_reviews.jsonl,arithmetic_checks.json,diagnostics.json,results.json,usage.json,acceptance.json}`、`data/processed/open-necessity-dev20-v1/`

### 2026-11-12：复查稳定性并测试证据verify闸门
- 做了什么：新增 `evidence_verification.py`、`verify-necessity-evidence-v1` 与 `verify-open-evidence`，12项/16 HTTP上限（5正确、2边界、5合成破坏）。
- 结果与决策：12次HTTP，17,445 tokens，60.853秒；5正确全保留、5坏证据全拒、2边界宽读全放行。`pytest` 111 passed。
- 失败与限制：不能宣称verify解决语义边界；同题独立重复性与真实题泛化仍未验证；N04总闸门拒绝正确但 `question_preserved` 仍pass而理由指时间冲突；同基座新会话不是独立金标准。
- 产物：`outputs/necessity-verify-dev-v1/{probe_report.md,items.jsonl,records.jsonl,calls.jsonl,results.json,usage.json,acceptance.json}`、`data/processed/necessity-verify-dev-v1/items.jsonl`

### 2026-11-12：真实原题20题与评估器实际修复
- 做了什么：按用户“必须修复而非只记录”指令，完成材料附图加载、共同K、假想补全与唯一解分支、显式证明结构、选项字母/文字检查、fail优先及最终合并，在同一冻结20原题重跑。
- 结果与决策：最终N=20：0确认双必要、17非双必要、3未决；原题有效性18确认/2未决；未决赋值边界0–15%。新增92 HTTP、130,309已知tokens、617.399秒；含此前共174请求、217,173已知tokens。125检查通过。
- 失败与限制：39正常分析+1中断；36份证明核验35正常+1超时；剩余季节量/整体量反例及两条新误放行经委托复核隔离，未靠重采样救结果；边界不是总体置信区间；分析与verify同基座，Codex知参考，不称独立金标准或自动稳定；2次失败/中断用量及总费用未知。
- 产物：`src/egvqg/{dataset,assessment_rules,open_necessity,evidence_verification}.py`、`configs/necessity_assessment_v2.json`、`outputs/real-scienceqa20-v2/`、`outputs/real-scienceqa20-verify-v2/`、`outputs/assessment-repair-v2/`

### 2026-11-12：默认评价模型切换gpt-6-luna
- 做了什么：用户选择接受模型评估的可记录偏差并指定更低价的 gpt-6-luna，仅切换本地默认模型名，用8份冻结旧证明单独run检查。
- 结果与决策：8项符合预先逐项裁定、零解析/技术失败（旧5.6同输入7项符合）；8 HTTP、14,248 tokens、115.842秒。不追加审核层、不静默混合新旧模型结果。
- 失败与限制：该开发检查非总体准确率或独立金标准，不宣称换模型解决所有偏差；费用未返回。
- 产物：`outputs/necessity-verify-gpt6-dev8/{probe_report.md,comparison.json,calls.jsonl}`

### 2026-11-12：必要性定义与论证文档（Claude，仅文档）
- 做了什么：新增 `docs/necessity_formal_argument.md` 以数学形式写出开放必要性评估器依据，并只读检查 `real-scienceqa20-v2` 与 `necessity-verify-gpt6-dev8`。
- 结果与决策：35次已完成核验中 `question_preserved` 与 `retained_preserved` 从未判fail，而三次已知误放行（R08/I、R14/T、旧R16/T）恰是这两类约束被破坏；误放行与证明长度无明显关系（65与453字符，中位126）。向用户提出65次调用的植入缺陷对照实验。
- 失败与限制：该实验尚未获批、未实现、未运行；未调用模型，未改代码。
- 产物：`docs/necessity_formal_argument.md`

### 2026-11-12：预注册保留约束诊断（用户固定设计，已完成后停止）
- 做了什么：按用户固定设计，对27份证明（8干净原样、每份两种合成单缺陷、3历史缺陷）跑 V0 与分拆抽取+核对，设计与读数在首调用前封存。
- 结果与决策：更正长度口径（36证明中位126.5、35成功核验中位127）；65实际HTTP（27 V0+11抽取+27核对，上限70），已知159,370 tokens，1,589.276秒；V0拒18/19缺陷，合取拒18/19，增量0；合取误拒5/8干净证明；简短/铺陈各漏0/8。满足“保留当前设计”与“不采用”条件；H-density按规则暂缓，H-frame未决。`pytest` 128 passed。
- 失败与限制：旧R16/T仍被V0放行，V1超时，操作上拒绝不计作识别缺陷；1抽取+5核对在3072上限未完成、1核对JSON无效、2核对前缀ID不符合约、3继承抽取失败、1超时；R04/T、R04/I、R13/T误拒干净证明，R07/I与R01/T技术失败；构造者Codex知标签，非独立人类判断；本轮停止，采纳由用户决定。
- 产物：`outputs/necessity-preservation-diag-v1/{seal.json,manifest.json,proof_set.jsonl,construction.json,calls.jsonl,records.jsonl,extractions.jsonl,usage.json,results.json,acceptance.json,probe_report.md}`

### 2026-11-12：本地Qwen3-VL官方test三组真实分析（用户选定规模，已完成）
- 做了什么：用户选择3组端到端，从官方test 1257组按 seed20261112 抽 1139/5000/20722，同材 plain/strong 各1次共6名额，复用用户指定 Video-skill-transfer 环境，新增 `local_inference.py` 与 `local-experiment-run`。
- 结果与决策：最终4有效/2有效性未决，0确认双必要/3非双必要/3未决；每方法N=3，BDR均0，plain另1未决、strong另2未决；30次本地推理、30次网关HTTP，47,057 tokens（本地7,680、网关39,377）、531.096秒。`pytest` 130 passed。
- 失败与限制：项目 `.venv` 无ML依赖，安装失败见 `environment_setup.log`；1份作答合同失败（Q04/0，answer=null而abstain=false）；磁铁普通题I理由颠倒极性，自动1改0；Q03/T与SiC两份I见证保持未决；不能估计总体准确率或给方法排名；Codex审核为非盲 user_delegated_ai；未扩30组。
- 产物：`outputs/qwen3-vl8b-test3-v1/`（`probe_report.md`、`review.html`、`analysis.json`、`metrics.json`、`usage.json`、`score_review.jsonl`、`acceptance.json`）、`src/egvqg/local_inference.py`

### 2026-11-12：baseline自然生成test30的问题与答案评估（已完成）
- 做了什么：用户纠正定位为评估现成baseline自然生成的Q是否含不可省略的双模态信息；用 Qwen3-VL-8B、plain-v1，排除已用1139/5000/20722后从test池取30组，本地30生成+120作答，gpt-6-luna做参考/评分/分析/verify。
- 结果与决策：Q19、Q20 两处具名更正后最终23有效/1无效/6有效性未决；合取后0确认双必要、17有效非双必要、1无效、12未决，N=30；13题T+K充分、8题I+K充分、4题重合；BDR为0通过/21不通过/9未决；120答题61作答、55拒答、4合同失败；6处评分更正（Q09/TI、Q17/T及TI、Q19/I由0→1，Q26/TI及I由1→0）；150本地推理+178HTTP共328调用、237,219 tokens、2,201.458秒。`pytest` 130 passed。
- 失败与限制：Q20候选参考1次未完成、Q15/I方向证明空答案结构失败，不补跑；Q16证明被verify拒，不翻为相反D；0确认不代表真实比例为0，未决界0–40%非置信区间；Q01图片曾出现；只支持本批baseline多数题至少一种模态可省略，不能推出所提方法优越，非盲独立人类；不自动扩量。
- 产物：`outputs/qwen3-vl8b-baseline-test30-v1/`（`probe_report.md`、`review.html`、`analysis.json`、`metrics.json`、`usage.json`、`acceptance.json`、`scope_amendments.jsonl`、`score_review.jsonl`）

### 2026-11-12：baseline忠实性诊断与固定分母派生复评
- 做了什么：按用户要求新增离线 `faithfulness-review`，将已证明的输入不支持、错误预设、比较范围不清记为生成问题缺陷，全部30题复核。
- 结果与决策：Q01仍无效；Q02/Q22、Q18/Q25/Q29、Q20 的6道有效性未决改V=0；结果23有效、7无效、0有效性未决；17有效非双必要、0确认双必要、7无效、6有效但必要性未决，N=30；派生BDR 0通过/27不通过/3未决。`pytest` 136 passed。
- 失败与限制：评审为 Codex 已见旧输出的非盲AI复核；参考错误或模型间答案不一致不能单独判Q错误；不据BDR声称真实双必要比例为0；边界题仍受K/教材简化约定影响；全1257组按用户要求继续暂停。
- 产物：`outputs/qwen3-vl8b-baseline-test30-faithfulness-v1/`（`probe_report.md`、`reviews.jsonl`、`records.jsonl`、`metrics.json`、`results.json`、`usage.json`）

### 2026-11-12：将可回答性与双模态必要性分开整理
- 做了什么：按用户要求把“可回答且图文缺一不可”的确认题数作为阅读重点，两类诊断分列展示。
- 结果与决策：可回答性/忠实性23通过、7不通过、0待裁定；23道可回答题必要性0两者都必需、17至少一种可省略、6待裁定；目标验收0达标、24不达标、6待裁定（N=30）。`pytest` 136 passed。
- 失败与限制：6题仅必要性待裁定、答案已核验；不能据0确认推断真实达标数必为0；全1257组继续暂停；修正JSONL导出格式的技术更正记在manifest。
- 产物：`outputs/qwen3-vl8b-baseline-test30-separated-v1/`（`probe_report.md`、`assessment_table.jsonl`）

### 2026-11-12：梳理当前证据与计划，定向API分析及全test API baseline启动
- 做了什么：用户选择“API生成并回答全部1257组新问题”；新增 `baseline-analysis-init/run/report` 对原7道可答性不通过与6道必要性未决题跑定向分析，并新增 `api-baseline-init/run/report` 启动 `luna-api-baseline-test1257-v1`（gpt-6-luna、plain-v1，3→30→1257）。
- 结果与决策：定向分析30次HTTP、64,480 tokens、1核验跳过；12份证明复核支持8/未决3/不支持1；Q05甜味遗传比例题双必要，Q13、Q14可由保留文本独立回答，Q07/Q16/Q26待裁定；Q02/Q22重开为可答性待裁定；当前30题为23可回答/5明确失败/2条件争议；可答题中1双必要/19非双必要/3待裁定；目标1达标/24不达标/5待裁定。`pytest` 140 passed。
- 失败与限制：不接受API核验通过但范围/答案粒度有争议的两份证据，不由失败推相反D；同模型自核验与非盲AI审核非独立人类金标准；全量运行进行中（已完成6组、HTTP79、57,759 tokens，会变），不可把未完成全量当最终结果；费用未知。
- 产物：`outputs/qwen3-vl8b-baseline-test30-api-analysis-v1/`、`luna-api-baseline-test1257-v1`

### 2026-11-12：按用户要求简化baseline、图片硬限与16路异步，跑完后再评估
- 做了什么：用户要求“baseline简单、控制图片大小、异步16线程、跑完再评估”，停止旧 `luna-api-baseline-test1257-v1`，新增 `simple-baseline-init/run/report` 与 `simple-baseline-eval-init`，全1257组生成+TI回答后再启动 `luna-api-baseline-test1257-eval-v1`。
- 结果与决策：`luna-api-baseline-test1257-simple-v1` 生成1256道Q、1237次作答、18次拒答、1生成失败、1回答失败；149次旧调用严格匹配复用，新HTTP2186、缓存复用178、884,653 tokens；图片硬限长边1024、100万像素、512KiB，1257张实际输入最大边760、484,943字节全合规。评估最多5次/题、6285次HTTP，任一模态充分解法通过即停该题剩余方向。`pytest` 147 passed；峰值16在途。
- 失败与限制：费用未知；旧2路/四条件流程已停止；评估进行中（129/1257），同模型自评仅标 automatic 暂定，不以1237条作答等同1237题正确或有效。
- 产物：`outputs/luna-api-baseline-test1257-simple-v1/`、`outputs/luna-api-baseline-test1257-eval-v1/`（`execution_acceptance.json`、`image_profile.json`、`post_batch_evaluation.json`）

### 2026-11-13：持续监控至全1257组统一评估完成并核对

- 按用户要求持续监控`luna-api-baseline-test1257-eval-v1`至结束，30秒一次只读健康快照，记录完成量、在途请求、错误、预算与进程状态；没有新增模型调用、重试或修改标签。监控见run内monitor.py、monitor.jsonl、monitor_status.json，最终state=completed、pending=0、active_requests=0，评估进程已退出。
- 最终自动暂定可答性：1100通过、118不通过、39待裁定；1100道可回答题中64图文都必需、976至少一种模态可省略、60必要性待裁定。核心固定N=1257为64通过/1094不通过/99待裁定，64/1257≈5.1%；99项明确分成39可答性与60必要性，不混称不能回答。原TI答案支持另列1088通过/111不通过/58未决或无可评回答，不把作答次数当正确率。
- 评估实际HTTP4179（上限6285）、缓存复用350、已知6,612,095 tokens；11份解析/未完成输出、6次技术失败全部保留，部分失败用量及费用未知。连同已完成的简化生成/回答，两阶段新HTTP合计6365、已知7,496,748 tokens，不重复记入149次旧调用消费。
- 完成全1257记录、4529次实际输入重建核对：源文件哈希、Q/原答案绑定、保留模态隔离、空历史、每角色单次尝试、逐题≤5及总预算、失败不翻转相反必要性、充分解法短路均通过。验收acceptance.json同时记录最终报告哈希。此检查只确认工程完整性和规则传播，不是独立语义正确率；64个自动候选仍待具体约束/K/答案粒度复核。
- 原位整理probe_report.md为可答性、必要性、原回答支持三张分表，主读数固定1257；README、实施路线与当前状态更新为全部完成，未新建平行总结。旧运行、冻结协议/K/历史指标和原Q/回答不变，147离线检查结果仍有效，未为日志排版重复跑测试。下一步优先语义复核64候选、99待裁定项及固定前30典型案例；本轮没有启动新生成、训练或方法比较。


### 2026-11-13：必要性评估器的稳定性试验（小规模，未并入流水线）

- 目的：针对全1257组自动结果中64个“双必要”候选做复核，同时试验让评估更稳。对64候选加26个按模板分层抽取的“可省略”对照题（seed 20261113），共90题。不改定义、冻结协议、K、BDR公式及历史结果；试验脚本在`tmp/`（v3_*.py），尚未进入`src/`，未补测试。
- 发现：luna单独判定的64个正例精度偏低。haiku盲测（看不到任何答案）发现16题单一模态即可答出；18道交换午餐题的“移除图像”反例违反题干预设；纠正了本人早先把`analysis_T/analysis_I`移除方向标反的错误（T=移除文本，I=移除图像）。haiku的“答不出”不是必要性证据（对称不变性类漏判），只用于降级。
- 方法试验：嵌入余弦判答案等价失败（AUC 0.65，阈值被推到0.99），产物改名`*_embedding_rejected.json`；逐对LLM等价判官受粒度影响波动；最终采用题干槽位模式加规范化取值比较，原子槽位使luna与deepseek取值一致率由74%升到83%–87%。模型组：gpt-6-luna、网关`DeepSeek-V4.1-Flash`（大小写与连字符须一致，能看图，推理关不掉、约3倍慢）、`gpt-5.6-terra`仅作仲裁。本地GPU有人使用，用户决定不用，已停止误启动的Qwen进程。
- 结果：整条流水线独立跑两遍，三组合并后两遍归属一致79/90（正例56/64，对照23/26）；两遍都确认双必要29、都否定13、标签不一致8；26个对照题无一被确认为双必要。稳定的系统性误差：答案为1/2的对称方格题（5题误放行）和交换午餐类（5题粒度争议），稳健下限约19题（1.5%）、含争议29题（2.3%），非对1257题的估计。用户裁定误差率够用，不再追加不变性枚举步骤；对称类写入`docs/necessity_formal_argument.md`第9节作为定义边界。
- 未解决：未在1257题上整体运行；对称类误放行仍存在；deepseek约5%调用失败按规则记未决；校准与比较用的仍是luna与同批模型判断，非独立人类金标准。
- 产物：`outputs/necessity-v3-pilot-v1/`（results_dual.jsonl、rule_layer_tri.json、rule_layer_*_rejected.json及各run日志）、`tmp/review64/`（候选包与haiku盲测）。用量未单独统计。


### 2026-11-13：全1257组必要性评估器单次运行（luna + deepseek，terra仲裁）

- 按用户要求只做ScienceQA、异步并发，对全1257组原baseline问题与原图文回答重跑新版必要性评估；可答性沿用原评估（118不通过、39待裁定、1099通过），本次未重算。不改定义、冻结协议、K、历史运行与原结果；试验代码仍在`tmp/`并快照到`code_snapshots/`，未并入`src/`、未补测试。
- 方法按`docs/necessity_formal_argument.md`第9节：两个模型组（gpt-6-luna、网关`DeepSeek-V4.1-Flash`）各做单模态充分性筛查（题干预设视为成立）和原评估器所给反例的复算，luna仅从题干生成原子槽位模式，两组各自抽取槽位取值，取值相同才算同一答案；两组未拿定的243题由`gpt-5.6-terra`按同样任务三取二仲裁。32路线程并发、每次调用最多重试一次、HTTP硬上限16000。
- 结果（固定N=1257，自动暂定）：通过（可答且图文缺一不可）38（3.0%），不通过1053（问题不通过118 + 至少一种模态可省略935），待裁定166（可答性待裁定39 + 无问题或回答1 + 必要性待裁定126）。原v2的64个正例中38个保持、14个被否定、12个变待裁定；原976个“可省略”中97个现为待裁定（两组不一致不硬判）。必要性待裁定126中82题是原评估器没有给出某方向反例（覆盖限制），44题是模型组不一致或槽位抽取未决。
- 限制：单次运行（90题两遍小试验归属一致率约88%，整体未重复）；已知误放行类为答案1/2的对称方格（约5题）和有粒度争议的交换午餐（8题）；图注型、矿物各1题被判双必要，前一轮试验曾判可省略，显示运行波动；只有原评估器给出两方向反例的约74题可能被确认；被评问题由luna生成且评估含luna，非人工金标准，未对通过题人工复核；deepseek槽位抽取失败43题，112条调用记录最终失败，均记未决。
- 用量：成功调用9659次（luna 4987、deepseek 3782、terra 890），已知词元7,430,600，费用未返回。
- 产物：`outputs/necessity-v3-full-v1/`（`probe_report.md`含三张分表与通过题清单、`final_items.json`逐题结果、`final_report.txt`、`results.jsonl`全部原始调用、`items_full.json`输入、`code_snapshots/`）。没有跑pytest；本次没有新增或修改`src/`代码，所以未重跑既有测试。下一步由用户安排（如对38个通过题与待裁定题抽查、整体重复运行评估稳定性、并入`src/`）。


### 2026-11-13：对全1257组评估器结果的人工抽查（claude-code，非盲，user_delegated_ai）

- 用户授权由本会话抽查。逐题读材料、问题、参考答案并查看图像，按答案可辨识性判据和K判断；记录在`outputs/necessity-v3-full-v1/audit.jsonl`并写入`probe_report.md`第八节，不改评估器输出、不改定义。
- 38道通过题：确认双必要19（非对称潘尼特方格18 + 磁铁Q1042）；偶然不变性7（2:2方格，答案1/2，按定义D_T=0，属定义边界）；误放行12（交换午餐8题——文本写明想要什么，加题干预设即可由文本定答案；Q0043；Q0584、Q0738——图像表格已给最终温度；Q0806——答案是文本性质列表）。通过题精度约50%（偶然不变性计入则68%）。
- 漏判4：Q0435、Q0699、Q0949（被两组按外部知识当作图像单独可答）、Q0646。抽查范围含全部8道非双必要潘尼特题、2道非双必要磁铁题、分层抽取的27道可省略和15道待裁定；Q0484、Q1130未看图。估计真正双必要约23/1257（1.8%），偶然不变性另约10题；935/126的抽样比例低，漏判可能更多。这是估计，不是全量人工标注，也非独立人类金标准。
- 未做：未改评估器或规则、未重跑、未补测试。误放行集中在交换午餐（预设未用于充分性）、温度表（题干重述图中信息）、单模态转抄，偶然不变性和外部知识为另两类系统性误差，重复运行不会消除。

- 最终口径改为两类（用户裁定“只有两个选项”）：已核实双必要23/1257（1.8%，下限）与其余；待裁定166题并入“其余”，仅作诊断。偶然不变性7题另列边界，计入为30（2.4%）。已写入`probe_report.md`第九节与`docs/necessity_formal_argument.md`第9节。更正：重复运行不能消除系统性误差，确认集合应全部复核。


### 2026-11-13：评估器并入src、文档梳理

- 把试验脚本并入统一代码库：新增`src/egvqg/necessity_ensemble.py`、4个提示模板（`prompts/ensemble-sufficiency-v1.txt`、`ensemble-witness-v1.txt`、`answer-slots-schema-v1.txt`、`answer-slots-extract-v1.txt`）、CLI`necessity-ensemble-init/run/report`和`tests/test_necessity_ensemble.py`（30项）。判定逻辑为纯函数；槽位抽取记录存下所问候选答案列表，补跑失败调用后不会与已缓存抽取错位；失败调用记录但不缓存、恢复时补跑；`--limit`的部分运行把未评估题单列，不混入未决。
- 验证：`.venv/bin/python -m pytest -q -o addopts=''` 177通过（原147 + 新30）；对9处关键逻辑做突变检查，8处被测试发现，1处（见证答案与完整答案相同的检查）在槽位相等传递的前提下逻辑冗余，已删除而不是补测试；用`outputs/necessity-v3-full-v1/results.jsonl`做零调用重放，新模块与旧脚本在1099题上的B、D_T、D_I零差异（`code_snapshots/compat.py`）；真实模型预检`necessity-ensemble-smoke-v1`，10题、84次HTTP、约6.5万词元，费用未返回。预检中发现并修复了新子命令漏加`--run-id`的CLI问题（现有parse测试覆盖）。
- 文档：新增`docs/references.md`（数学命题索引、论文与数据来源，均取自`story.md`、`VQG_Proposal.md`、README的原有引用，未增补标题；3个arXiv编号和SynMQG链接沿用原文，未联网核对）；README改为简短当前状态、文档地图表和必要性评估入口；`docs/evaluation_design_v2_draft.md`、`docs/implementation_plan.md`加状态说明；`docs/evaluation_protocol_v1.0.md`未改。相对链接检查无断链。改动前备份在`outputs/_doc_backup_2026-11-13/`。
- 未做：`tmp/`里试验脚本和`tmp/review64`的清理或移动（权限未批准，用户决定），脚本已复制进`outputs/necessity-v3-pilot-v1/code_snapshots/`和`necessity-v3-full-v1/code_snapshots/`；PROGRESS.md（约145KB）、README后半的历史命令段、`implementation_plan.md`按时间的历史段未压缩；旧的评估/生成模块与约60个CLI子命令未清理（冻结运行的哈希依赖它们）；仓库一个提交且全部文件未跟踪，未做git提交或推送。


### 2026-11-13：卡3本地Qwen3-VL-8B跑全1257组baseline、评估器只跑可能被确认的题

- 用户要用卡3在ScienceQA test上跑本地Qwen3-VL-8B（Qwen3.5-9B暂缓：其`model_type`为`qwen3_5`，共享venv的transformers 4.57.6不识别，需另装更新版，用户选择以后再说）。只使用卡3（`CUDA_VISIBLE_DEVICES=3`），共享venv`Video-skill-transfer/.venv`可用于Qwen3-VL-8B（torch 2.8.0+cu128、transformers 4.57.6、Blackwell sm_120），该venv无pip、无vllm。
- 新增`src/egvqg/local_vision.py`（`BatchingProvider`动态批处理 + Qwen3-VL后端，贪心、bf16、sdpa）和`simple-baseline-run --local-model-path/--device/--max-batch/--max-new-tokens`，5项测试；批大小不进模型身份。预检16题21秒（含模型加载）；全量`qwen3vl8b-baseline-test1257-simple-v1`：1257题全部生成（0生成失败）、回答1119、拒答75、内容合同失败63；与luna baseline相比回答失败多（63对1）、重复问题字符串多（414对237）。输出尚未评估。
- 成本分析（对全量评估器结果拆分）：9,888次调用、743万词元，平均每题约9次；唯一可能被确认的74题（两方向都有反例）只占994次调用、15%词元，其余1025题（93%）占90%调用、85%词元却不可能被确认。单模态充分性筛查把94%的题判为单模态可答，对审计确认的真双必要误砍0/19（含漏判共1/23），并去掉38个确认项里的6个。上游原评估器本身另有661万词元。
- `necessity-ensemble`默认只评估两方向都有反例的题（`--all-items`可全量诊断），其余记为“无双向反例，未运行”并计入未确认；31项测试。旧run的manifest无该设置，按全量处理，结果不变。
- 未做：对Qwen baseline的评估（拟采用“有效性 → 单模态筛查 → 仅幸存题做反例与评估器”的顺序，需先改评估流程）；Qwen3.5；本地模型用作筛查器的误砍率检验。


### 2026-11-13：Qwen3-VL-8B baseline的级联评估与必要性评估（独立参考答案）

- 被评对象是本地Qwen3-VL-8B的全1257组baseline（luna那批只是调试参照）。评估流程改为级联：有效性（1次）→ 单模态充分性筛查（2次，任一模态能唯一作答即停）→ 只对幸存题生成两方向反例（不再逐条核验）→ 必要性评估器只评两方向都有反例的题。`simple-baseline-eval-init --cascade`，默认关闭，旧流程不变；评估一次发现并修复了级联分支里用`out`盖住输出目录的变量覆盖bug。
- 评估结果（`qwen3vl8b-baseline-test1257-eval-v1`，3,058次HTTP、432万词元）：有效962、不通过268、待裁定27；原回答有依据741、无依据352、未决164；筛查砍掉894题（占有效题93%）。无效题失败码：参考无依据206、缺决定性前提185、预设无依据166、范围含糊61。
- 必要性评估器新增`reference`设置：默认由两个评估组用完整材料独立作答，一致才作参考；`--baseline-reference`保留旧行为。原因：Qwen回答352道无依据、61题没有回答，用它作参考会漏评并误判。第一版`qwen3vl8b-necessity-v1`（用Qwen回答作参考）保留，第二版`qwen3vl8b-necessity-v2`（765次HTTP、约41万词元）为准。默认只评两方向都有反例的题（`--all-items`才全量），报告拆出“源评估已判单模态即可”的数量。
- 结果（固定N=1257，自动确认28，抽查后）：已核实双必要19（1.5%，非对称潘尼特14加是非型交换午餐5）；偶然不变性6；误放行3（Q0232、Q0273、Q0367）；另5道同型是非题评估器未决、据题型判断很可能漏判但未看图，单列不计。其余1238中单模态即可作答896（71%）、无两方向反例16、评估器未决22、无效268、有效性待裁定27。对照luna那批（约1400万词元）API词元约便宜66%。审计记录`outputs/qwen3vl8b-necessity-v2/audit.jsonl`，报告`probe_report.md`。
- 限制：只抽查了确认的28题，对“其余”没有分层抽样，19是下限；被前置筛查砍掉的894题若有误砍不会被找回，该筛查对真双必要的误砍率只在luna那批验证过（0/19）；偶然不变性评估器仍然识别不出。
- 验证：`.venv/bin/python -m pytest` 109通过（新增级联2项、独立参考3项）；新增独立参考与单模态筛查的材料隔离检查；未做突变检查。


### 2026-11-13：Qwen baseline前置筛查的抽查（存在性命题）

- 目的是确认“多数可回答的问题，图和文至少有一个多余”。从Qwen评估中被前置筛查砍掉的894题分层随机抽40题（种子20261113），拼图后由haiku盲答（看不到评估器结论），不一致的题由我看图裁定；记录`outputs/qwen3vl8b-necessity-v2/screen_audit.jsonl`，结论写入同目录`probe_report.md`第六节。
- 结果：haiku盲答直接复现35题；其余5题（#3、#25、#34、#31、#35）经看图裁定评估器正确（haiku在拼图里读小字和箭头较弱，#31、#35的箭头未在拼图分辨率下核实）；评估器判错0题。“单模态即可作答”抽样40/40，Wilson 95%下限约91%；被误砍的真双必要题上界约8%（三之法则）。折算：962道有效题中至少约85%的图与文至少有一个多余。
- 限制：只检查“是否不是双必要”，没逐一核对被判充分的具体模态；样本40；haiku加我的裁定，非独立人类金标准。
