# 参考资料索引

本文只汇总项目文档里已经引用的外部文献和数据来源，以及项目自己的数学定义入口。称呼与链接取自原文，未增补标题或作者；原文位置标出，便于回查。

## 1. 数学定义与论证（项目自有）

全部在 [necessity_formal_argument.md](necessity_formal_argument.md)，不依赖外部文献。

| 内容 | 位置 |
|---|---|
| 记号、有效性 $V$、相容补全 $C_T,C_I$、假设 A1 | 第1节 |
| 模态必要性 $D_T,D_I$（定义1）、二分（引理1）、双必要 $B$（定义2） | 第2节 |
| 充分解法（定义3，命题1：充分解法 ⇒ $D=0$）、见证（命题2：见证 ⇒ $D=1$）、失败不传递（命题3） | 第3节 |
| 答题器行为不是必要性证据（命题4） | 第4节 |
| 缺失不等于必要、不变性（命题5）、题干提供的事实不能当见证变量（命题6） | 第5节 |
| 最小答案要点与整合性（命题7、8） | 第6节 |
| 三值汇总、两类核验错误的不对称（命题10）、多核验器合取（命题11）、固定分母下的界（命题12） | 第7节 |
| 论证没有解决的事 | 第8节 |
| 评估实现的三条操作约定：预设闭包、题干槽位定义答案等价、两组一致三取二；偶然不变性边界；两类报告口径 | 第9节 |

评估规格与协议另见：[evaluation_protocol_v1.0.md](evaluation_protocol_v1.0.md)（已冻结，不修改）、[evaluation_design_v2_draft.md](evaluation_design_v2_draft.md)（设计草案，历史）。

## 2. 论文

### 2.1 捷径学习与视觉依赖

| 文献（原文称呼） | 链接 | 原文位置 | 原文中的用法 |
|---|---|---|---|
| 捷径学习 | <https://arxiv.org/abs/2004.07780> | story.md:50 | 引言从机器学习中的捷径学习共性问题出发 |
| VQA v2 | <https://arxiv.org/abs/1612.00837> | story.md:50、story.md:102 | 语言先验能掩盖视觉能力不足；以互补图像检查答案对视觉内容的依赖，作为视觉依赖诊断的先例 |
| Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models | <https://arxiv.org/abs/2607.26326> | VQG_Proposal.md:10、story.md:50 | 感知—利用差距：区分视觉信息的编码与可靠利用；本研究的直接动机 |
| VideoQA视觉依赖审计 | <https://arxiv.org/abs/2606.30220> | story.md:102 | 比较有无视觉输入时的表现 |

### 2.2 生成评测

| 文献（原文称呼） | 链接 | 原文位置 | 原文中的用法 |
|---|---|---|---|
| 生成文本评价实践研究 | <https://arxiv.org/abs/2202.06935> | story.md:68 | 讨论表面质量、代理指标与实际目标之间的失配 |
| ConVQG的评测 | <https://arxiv.org/html/2402.12846v1> | story.md:72 | QG评测常用自动文本指标并结合人工偏好比较，不直接检验答案需要哪些材料 |

### 2.3 视觉问题生成（VQG）

| 文献（原文称呼） | 链接 | 原文位置 | 与本文的区别（原文） |
|---|---|---|---|
| 早期VQG | <https://aclanthology.org/P16-1170/> | story.md:74 | 研究自然提问 |
| IM-VQG | <https://arxiv.org/abs/1903.11207>；<https://openaccess.thecvf.com/content_CVPR_2019/html/Krishna_Information_Maximizing_Visual_Question_Generation_CVPR_2019_paper.html> | story.md:74、VQG_Proposal.md:394 | 最大化图像、预期答案与问题之间的互信息；本文不预指定答案 |
| ConVQG | <https://arxiv.org/abs/2402.12846> | story.md:74、VQG_Proposal.md:395 | 双重对比目标促进图像扎根与文本引导；本文把联合利用落实到答案的信息需求 |
| SynMQG | <https://shuangyinli.cn/publications.html> | story.md:74、VQG_Proposal.md:396 | 解耦与互信息优化的协同多模态问题生成；本文以作答必要性组织证据选择与评价 |
| VisPlay（自博弈问答合成） | <https://arxiv.org/abs/2511.15661> | VQG_Proposal.md:397 | 出题与答题协同演化；本文不以一致性或难度作为产出判据 |

## 3. 数据来源

| 数据 | 链接 | 原文位置 |
|---|---|---|
| ScienceQA 官方数据说明 | <https://scienceqa.github.io/> | VQG_Proposal.md:247 |
| ScienceQA 官方仓库（下载与字段） | <https://github.com/lupantech/ScienceQA#-download-the-dataset> | README.md、VQG_Proposal.md:256 |
| MultiModalQA（AllenAI） | <https://github.com/allenai/multimodalqa> | README.md |
| WebQA（WebQnA） | <https://github.com/WebQnA/WebQA> | README.md |

## 4. 说明

- 本文件不独立判断文献内容；改动原文引用时请同步这里。
- 三个arXiv编号（2607.26326、2606.30220、2511.15661）和SynMQG主页链接沿用原文，本次整理没有联网逐条核对，投稿前应核对标题、作者与版本。
