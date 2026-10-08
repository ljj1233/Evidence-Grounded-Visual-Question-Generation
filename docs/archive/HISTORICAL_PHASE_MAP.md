# 阶段沿革与最终 checkpoint 的关系

2026-10-08；以 VQAv2 Full Ours step256 和 Full Ours+DPO step207 为目标。`superseded` 仅表示该实验方案不再是指定 checkpoint 的生成方案，**不是清理许可**。文件内 helper、数据或评估实现仍可能被 CURRENT 调用；逐文件判定见 [SCRIPT_CLASSIFICATION.tsv](SCRIPT_CLASSIFICATION.tsv)。

## 1. 最终权重继承链

```text
public Qwen3-VL-8B-Instruct
 → Final SFT 50k / 2 epochs / 6250 steps
 → Phase5C D1 single-positive / 10000 records / 1250 steps
 → Phase5C G2 static UVFC + init-relative guard / step256 = Full Ours
 → unified offline DPO / selected step207 = Full Ours + DPO
```

**Phase3A、Phase4A RFT、Phase4B static/dynamic、Phase5A LatentCov、Phase5B Explore-Ground 的训练权重不是顺序串接到 Full Ours 的祖先。** 数据、helper 和方法设计被继承，不等于 checkpoint 被继承。证据：G2 config 的 init=D1；D1 summary 的 init=Final SFT。

## 2. 主编号阶段

| Phase / 分支 | 相比前一研究阶段改了什么 | 真实入口 / 定义证据 | 进入最终 Full Ours？ | 当前状态 / superseded |
|---|---|---|---|---|
| Phase1 SFT / 1.8诊断 | public base → image→question LoRA SFT；随后检查候选质量 | `scripts/vvqg/prepare_sft_data.py`、`src/vvqg/training/train_sft.py`、`configs/new_vqg/sft.yaml`；Final SFT training_summary | **是**，最终模型和 teacher/UVFC 的祖先；1.8诊断非 loss | SFT CURRENT；`diagnose_candidates.py` DIAGNOSTIC |
| Phase2.1 Flat calibration | 冻结 requirement-aware visual support verifier | `src/vvqg/rewards/support/{calibration,frozen_verifier,online_flat,structured_flat}.py`；`docs/VISUAL_SUPPORT_CALIBRATION_V1.md` | **是**，teacher 与 G2 都用 Flat；外部GPT评估另行独立 | Flat仍CURRENT；历史校准分析是 DIAGNOSTIC |
| Phase2.1B–E | dependency-aware aggregation、controlled minimal pairs、anchor-first验证、错误归因 | `docs/VISUAL_SUPPORT_DEPENDENCY_ABLATION_V1.md`、`VISUAL_SUPPORT_CONTROLLED_VALIDATION_V1.md`、`VISUAL_SUPPORT_CONTROLLED_VALIDATION_V2.md`、`VISUAL_SUPPORT_ERROR_ATTRIBUTION_V1.md` | 未替换最终 Flat；作为边界验证 | HISTORICAL/DIAGNOSTIC；dependency替代方案NO-GO |
| Phase2.2A / 2.2C | 信息选择 judge；后续 real/complement/blank VIC 检验 | `scripts/vvqg/visual_information_contribution.py`；`docs/INFORMATION_SELECTION_CALIBRATION_V1.md`、`VISUAL_INFORMATION_CONTRIBUTION_V2.md` | **否**，G2无VIC/answer-likelihood reward | HISTORICAL/DIAGNOSTIC；被UVFC研究方向替代 |
| Phase2.3 / 2.4 / 2.4C | 冻结评估协议、SFT泄漏审计、baseline矩阵、同骨干baseline设施 | `docs/EVALUATION_PROTOCOL_V1.md`、`SFT_LEAKAGE_AUDIT_V2.md`、`SAME_BACKBONE_BASELINES_SPEC.md`；canonical evaluate脚本 | 评估协议/公共基础设施仍使用；baseline权重非主模型祖先 | CURRENT_EVAL / 独立baseline；非整阶段废弃 |
| Phase2.5 / 2.5B | 离线reward排序、VIC失败归因 | `phase25_analyze.py/phase25_vic.py/phase25b_vic_attribution.py`；`docs/OFFLINE_REWARD_AUDIT_V1.md`、`VIC_FAILURE_ATTRIBUTION_V1.md` | 不进入G2 loss | DIAGNOSTIC；旧候选池被2.6审计利用 |
| Phase2.6 A | question-side visual dependence 路线 | `phase26_qdep.py`；`phase26_compare.py` | 否 | HISTORICAL/DIAGNOSTIC |
| Phase2.6 B / 2.6C | FAC风格SAE→real/blank视觉敏感度→undercoverage→VFC/UVFC；实现审计冻结 | `phase26_vfc.py`、`src/vvqg/rewards/visual_information/uvfc.py`；`docs/UVFC_IMPLEMENTATION_FREEZE.md` | **是**，sae.pt/weights.pt与公式仍CURRENT | B资产CURRENT；C审计DIAGNOSTIC；未被Dynamic/LatentCov替换 |
| Phase3A P1 / P2 | 固定 rollout 的组相对优化；P1只UVFC，P2 violation-first | `train_fixed_group_relative.py`、`configs/phase3a/{p1_uvfc_fixed_group,p2_uvfc_violation_fixed_group}.yaml`；`docs/FIXED_GROUP_RELATIVE_PILOT_METHOD.md` | 旧checkpoint否；`canonicalize_group/clipped_surrogate/build_trainable_model` helper **仍是依赖** | 训练方案由online G2替代；不可整文件判废 |
| Phase3A.2 P1strong / 3A.4 multiseed | 增强固定rollout训练预算/lr，后做seed43/44确认 | `configs/phase3a/p1_uvfc_stronger.yaml`、`p1strong_seed43.yaml`、`p1strong_seed44.yaml`；`docs/P1STRONG_MULTI_SEED_CONFIRMATION.md` | 否 | HISTORICAL，非G2初始化 |
| Phase3A.3 spatial attribution | 分析UVFC空间问题和feature分布 | `phase3a_spatial_{audit,qtype,uvfc_features}.py`；`docs/PHASE3A_SPATIAL_ATTRIBUTION.md` | 非loss | DIAGNOSTIC_ONLY |
| Phase4（总称） | 从固定rollout试验转向RFT及在线rollout研究 | 下列4A/4B代码 | 部分思想/数据/helper；没有独立“phase4最终trainer” | 须按子阶段判断 |
| Phase4A RFT | 冻结候选池中选UVFC-only或violation-first最佳问，再做SFT | `phase4a_select.py`、`configs/phase4a/*.yaml`；`docs/PHASE4A_UVFC_RFT_RESULTS.md` | 否，G2/D1不从RFT权重起步 | HISTORICAL / superseded as final method |
| Phase4A.1 | UVFC+Violation-RFT多seed确认 | `configs/phase4a/uvfc_violation_seed43.yaml`、`uvfc_violation_seed44.yaml`；`docs/UVFC_VIOLATION_RFT_MULTI_SEED.md` | 否 | HISTORICAL |
| Phase4A.2 | 加入1:1原人工问题SFT replay | `phase4a2_build_data.py/phase4a2_train.py`；`configs/phase4a2/replay_sft.yaml`；`docs/PHASE4A2_RFT_SFT_REPLAY_RESULTS.md` | 否；G2无replay CE | HISTORICAL |
| Phase4A.3 | replay方案seed43/44确认 | `configs/phase4a3/*.yaml`；`docs/PHASE4A3_RFT_REPLAY_MULTI_SEED.md` | 否 | HISTORICAL |
| Phase4A.4 / 4A.5 | 数据/模型/评估完整性审计；后续repository hygiene | `phase4a4_*.py`、`repo_cleanup_audit.py/repo_cleanup_execute.py`；`docs/PHASE4A4_FULL_INTEGRITY_AUDIT.md` | 非方法组件 | 审计DIAGNOSTIC；cleanup执行器UNKNOWN运维用途，不能运行 |
| Phase4B.0 | 改为每步新采样的online UVFC+Flat优化，可行性2000 records | `train_online_uvfc_grpo.py`、`configs/phase4b/feasibility.yaml`；`docs/ONLINE_UVFC_GRPO_RESULTS.md` | 旧run否；online机制被继承 | HISTORICAL feasibility |
| Phase4B.1 Static | Final SFT起步、固定SAE/v/u、10k训练manifest | `configs/phase4b/static_uvfc.yaml`、`src/vvqg/training/online_uvfc_grpo.py`、`phase4b_prepare.py` | 权重否；**manifest、Static机制、加载helper仍使用** | 旧训练入口superseded；公共依赖CURRENT |
| Phase4B.1 Dynamic | 与Static相比每200步更新u，v仍冻结 | `configs/phase4b/dynamic_uvfc.yaml`、`uvfc.dynamic_u` | 否；G2 static且refresh_every=0 | HISTORICAL；非最终选择 |
| Phase4B.2 | 与Static相比改成public base fresh LoRA初始化，仍用SFT anchor | `configs/phase4b/base_uvfc.yaml`；`docs/PHASE4B_BASE_UVFC_GRPO_RESULTS.md` | 否 | HISTORICAL initialization ablation |
| Phase4B.2b | 在base-init上进一步重建base-space SAE/anchor | `configs/phase4b/base_uvfc_baseanchor.yaml`、`phase4b2_build_base_vfc.sh` | 否；最终配置仍指向phase26/vfc | HISTORICAL；不能混同GQA base-init主线 |
| Phase5（总称） | 探索扩展问题空间、证据筛选、蒸馏，再接UVFC refinement | 下列5A/5B/5C | 最终取5C D1+G2分支 | 按子阶段判断 |
| Phase5A Prompt-CoT / Luna / Explore-4 | 对比推理式探索、外部生成器、exploration prompt K4 | `phase5a_prompt_cot_generate.py/phase5a_luna_generate.py/phase5a_explore_generate.py`；`docs/PHASE5A_LLM_VQG_BASELINES.md` | 非Full Ours teacher；是比较/副线 | HISTORICAL baseline experiments |
| Phase5A LatentCov | Static UVFC额外加frozen latent coverage reward | `train_online_uvfc_latentcov.py`、`configs/phase5a/static_uvfc_latentcov.yaml`；`docs/PHASE5A_EXPLORATION_UVFC_ANALYSIS.md` | 否；G2无此项 | HISTORICAL |
| Phase5B Direct-4 / S0,S1,S2 | 对比plain Direct-4与Explore-4；support-floor优先与Top2选择 | `phase5b_direct4_generate.py/phase5b_score_candidates.py/phase5b_select.py`；`docs/PHASE5B_EXPLORE_GROUND_DISTILL.md` | **Direct-4、评分器、S1 fallback仍CURRENT**；不是全部S0/S2被采用 | 部分继承；旧selector实验HISTORICAL |
| Phase5B Explore-Ground D/E | Explore-4+S1 teacher蒸馏，再沿旧Static guard做RL | `phase5b_build_teacher.py`、`configs/phase5b/{distill,explore_ground_uvfc}.yaml` | 否；最终D1来自Direct-4+S3 | HISTORICAL，superseded by5C |
| Phase5C-A S3 | 增加support_floor≥τ约束，在可行集内最大UVFC；无可行者fallback S1 | `phase5c_select.s3`；τ=.5；`docs/PHASE5C_EVIDENCE_CONSTRAINED_QSE.md` | **是**，teacher selection | CURRENT |
| Phase5C-B / 5C-8 / 5C-9 D1 | 相同teacher最佳问题重复两个epoch蒸馏 | `phase5c_build_targets.py/phase5c_distill.py`；D1 training_summary | **是**，G2 init/reference | CURRENT；不是multi-positive版本 |
| Phase5C-B D2 | 同预算，两个epoch使用不同正例（有时相同） | `targets_d2_multipos.jsonl`、`outputs/phase5c_distill_multipos/` | 否 | HISTORICAL ablation，未被选作G2 init |
| Phase5C-C G1 | SFT init，改用训练前校准的init-relative guard | `configs/phase5c/static_relativeguard.yaml`、relative trainer | 不是祖先；是G2 matched ablation | HISTORICAL comparison；同trainer仍CURRENT |
| Phase5C-C G2 | 相对G1仅改D1 init（连带reference和reward激活模型），静态UVFC与相对guard | `configs/phase5c/qse_relativeguard.yaml`、`train_online_uvfc_relative.py` | **是，Full Ours step256** | CURRENT checkpoint；当前源码guard已后续修正 |

## 3. Phase5C 之后的副线与修正

下列是后续实验，**不是自动替换了最终 Full Ours**。Phase编号只在各自报告内部有效，例如EC的Phase0/1/4不能当成主线Phase4。

| 分支 | 改动 / 证据 | 进入指定Full Ours？ | 状态 |
|---|---|---|---|
| coverage iteration A1/A2 | aligned Direct-4、增强候选池、gate；`reports/coverage_iteration_20261004T023610Z/FINAL_RECOMMENDATION.md` | 否；gate失败，未进入D1/RL | HISTORICAL/DIAGNOSTIC |
| guard patience patch | `DEGEN_STREAK=3` + `tests/test_guard_patience.py` | 当前代码是；历史step256不是该停止行为；未运行计划中的G2 1000-step rerun | CURRENT SOURCE CORRECTION / historical checkpoint discrepancy |
| EC-QSE / ECS | Evidence-Calibrated Selector、prompt alignment、阈值方案；`reports/ec_qse_revision_20261004T085451Z/FINAL_REPORT.md` | 否；selector revision failed；但 `ec_aligned_direct4_generate.py` 被DPO候选构建复用 | HISTORICAL实验 / CURRENT DPO helper |
| E1 joint soft evidence / E2 candidate-assigned slots | `ec_evidence_train.py/ec_slot_assignment_train.py` 调用evidence wrapper；`ec_e2_*`检查slot/decoder use | 否；最终adapter推理没有这些wrapper | HISTORICAL训练、DIAGNOSTIC审计 |
| VC-QSE / concept / oracle | `src/vvqg/vc_qse/` 与 `scripts/archive/vc_*`、`vc_support_verify.py` | 否；不是原Direct-4+S3 QSE | HISTORICAL/DIAGNOSTIC；本次未更动现有archive |
| G3 support-preserving / corrected PPO | `scripts/archive/g3_train.py`、`src/vvqg/training/g3_reward.py`：调整reward与old-logprob复用方式 | 否；不能把corrected PPO写进G2 | HISTORICAL训练 / DIAGNOSTIC gate |
| CER counterfactual | `src/vvqg/evidence_constrained/`、`scripts/archive/evidence_constrained_*`；`reports/evidence_constrained_full_20261005T165813Z/FINAL_REPORT.md` | 否；未产生进入主线的active CF-loss checkpoint | HISTORICAL / DIAGNOSTIC |
| Support-Delta | `src/vvqg/support_delta/`、`scripts/vvqg/support_delta_*`；已有pilot报告 | 否；即便独立branch有后续dev/holdout产物也不属于这两个目标的训练依赖 | HISTORICAL相对本目标；当前其他任务状态不由此推断 |
| image intervention / signal audits | `full_ours_intervention_*`、`cer_signal_audit_*` 等对现有模型做诊断 | 不改变权重 | DIAGNOSTIC_ONLY |
| GQA-native SFT/base-init链 | `configs/gqa_native/*`、`gqa_full_ours*_chain.sh`；`docs/GQA_NATIVE_FULL_OURS_METHOD_CONTRACT.md` | 不属于VQAv2两checkpoint祖先；有不同数据/init/guard版本 | UNKNOWN_REQUIRES_REVIEW / 独立活动方向；不标废弃 |
| 同骨干/公开baseline | B2/B3、SeeQ faithful、LOVA3、Genixer；各自config与脚本 | 不属于Full Ours训练；评估比较可复用 | 独立baseline；推理/比较CURRENT_EVAL，训练状态UNKNOWN |

## 4. DPO 单列：Post-training refinement

| 分支 | 相比上版 | 进入最终DPO step207？ | 状态 |
|---|---|---|---|
| DPO pilot / fresh confirmation | 小规模Flat pairs验证与新dev检查 | pilot权重否；共用 `src/vvqg/dpo_pilot/trainer.py` **仍当前实现** | HISTORICAL pilot，CURRENT trainer helper |
| Risk-aware DPO v2 | S/U/X、额外筛选或加权与relaxed guard | 否 | HISTORICAL |
| Formal original 6633 | aligned train9184 K4 + Flat分类/UVFC extremum生成6633对 | **数据是来源**；旧formal训练权重不是unified init | CURRENT data lineage / HISTORICAL旧训练 |
| batch8 controlled original | 相同原pairs改batch8控制实验 | 对比证据，权重否 | HISTORICAL |
| Matched M / CONTROL-M / TREATMENT-M | 固定chosen，同类型/句式/长度/TF-IDF匹配negative | **TREATMENT_PAIRS数据被复用2733对**；control/treatment权重否 | CURRENT pair builder / HISTORICAL权重 |
| multi-positive / evidence-mode H–N | 增加类型/positive与语义可行性分析 | 否 | HISTORICAL/DIAGNOSTIC |
| Unified 6633 | 原6633完整图集合，2733 matched +3900 fallback | **是**，Full Ours init/reference，unweighted DPO | CURRENT_DPO_REFINEMENT |
| unified full schedule step830 | 一epoch训练终点 | 不作为最终发布adapter；提供checkpoint trajectory | HISTORICAL unselected checkpoint；训练run本身CURRENT provenance |
| DEV256 step207选择、DEV744确认 | 从既有轨迹选较早checkpoint控制类型/多样性退化 | **是，最终冻结结果** | CURRENT_EVAL / selection provenance |
| weighted fallback Phase2 | 仅在Phase1无满意checkpoint时启用的计划 | 否；Phase1报告明确未运行 | NOT_RUN；不得写成最终loss |

## 5. 已知记录冲突与分类含义

1. `CURRENT_STATUS.md` 的早期“DPO被移除、GRPO未开始”不覆盖现存Phase5C/DPO产物。
2. Phase5C 文档开头的“Grounded Multi-Positive”是研究框架，最终G2配置选的是 **D1 single-positive**；D2属于未采用消融。
3. G2 step256的单步退化停止与当前源码patience3不同，历史精确源版本未冻结到当前checkpoint记录。
4. Phase4B/Phase3A文件中被import的helper仍CURRENT，不能按phase名字整体清走；已有归档脚本位置以TSV为准。
5. DPO final unified builder未定位；已对全部6633对验证其合并结果，不把评估`union`脚本误记为pairs builder。
6. 旧freeze说holdout sealed，后来的final_evaluation已经报告holdout；它们是不同时间的状态。

TSV类别：`CURRENT_MAIN_PIPELINE`=指定模型的训练/数据依赖（scope分开FULL_OURS与DPO_REFINEMENT）；`CURRENT_EVAL`=直接推理、最终/必要比较评估；`HISTORICAL_EXPERIMENT`=已定位的另一实验方案；`DIAGNOSTIC_ONLY`=核验、归因、可行性或错误诊断；`UNKNOWN_REQUIRES_REVIEW`=源码证据不足、运维、cache或独立活动方向。未知不等于废弃；HISTORICAL也不等于可删。
