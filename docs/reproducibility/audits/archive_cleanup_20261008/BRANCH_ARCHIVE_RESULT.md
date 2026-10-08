# 旧实验 / 分支整理结果

2026-10-08；依据用户附件与既有方法审计执行。

## 执行结果

| 项目 | 数量 / 结果 |
|---|---:|
| ENVIRONMENTS_TOUCHED | NO |
| DATASETS_TOUCHED | NO |
| CURRENT_CHECKPOINTS_TOUCHED | NO |
| CURRENT_FULL_OURS_PIPELINE_BROKEN | NO |
| CURRENT_DPO_PIPELINE_BROKEN | NO |
| CURRENT_GQA_PIPELINE_BROKEN | NO |
| CURRENT_BASELINE_PIPELINE_BROKEN | NO |
| FILES_MOVED | 131 |
| HISTORICAL_SCRIPTS_ARCHIVED | 42 |
| DIAGNOSTIC_SCRIPTS_ARCHIVED | 33 |
| CONFIGS_ARCHIVED | 20 |
| DOCS_ARCHIVED | 21 |
| REPORTS_ARCHIVED | 113 |
| REPORTS_ARCHIVE_MODE | COPY; 0 report sources moved |
| DUPLICATE_SCRIPTS_RELOCATED | 15 |
| UNKNOWN_FILES_TOUCHED | 0 |
| FILES_DELETED | 0 |
| PYTEST_PASSED | 155 |
| PYTEST_WARNINGS | 1 |
| RETAINED_CANDIDATES | 49 |
| HASH_UNCHANGED | True |
| ROLLBACKS_REQUIRED | 0 |
| FUTURE_OUTPUTS_INDEX_ONLY | 31 |

脚本42+33、配置20、文档21、重复脚本15合计131次MOVE。113份报告为COPY，不计入FILES_MOVED。所有移动和复制均逐项核对前后SHA256相同。未删除原有文件；未改方法源码、tests、环境或.gitignore；本次未执行git add/commit/push。

## 保护与依赖核验

- 171个CURRENT_MAIN_PIPELINE / CURRENT_EVAL / UNKNOWN原路径全部存在。
- 8,389项未移动的代码/测试/受保护产物及目录元数据（inode、size、mtime）一致；没有归档导致的保留依赖失效。
- 已扫描src/scripts/configs/tests/shared/baselines/third_party中的文本引用及模块名，按保留源的依赖闭包保护候选文件。完整边记录见BRANCH_DEPENDENCIES.tsv。
- .venv中15个当前主线脚本及10个核心模块import smoke通过；metrics环境中22个当前评估脚本import smoke通过。未调用训练main、模型加载或生成。
- 11个保留的shell入口bash -n通过；三份当前方法文档的75个本地链接均存在。
- NO/BROKEN检查指本次移动未引入路径或import回归，不代表重新训练或重新评测所有模型。

## 保留项

49个原候选因运行审计、现存tests或当前完整性/复现/baseline工作流依赖而保留；见BRANCH_ARCHIVE_PLAN.tsv中的KEEP。

既有PID182816运行/tmp/patch_hash_reports.py，依赖原文件清单。未中断该进程；其输入脚本和文档保留，历史报告按附件授权COPY。每个报告副本目录均含SOURCE_PATH.txt，指向原目录及原大型artifact引用。

所有当前GQA、faithful SeeQ、B2/B3、LOVA3/Genixer方向及UNKNOWN保持原位。configs/new_vqg/grpo.yaml被现存src入口引用、Phase3A P1/P2 configs和phase23_report.py/phase4a_select.py仍被tests读取，故保留。

## 测试

- 首次按仓库示例直接运行pytest：3个collection errors（未将src加入搜索路径），移动前即存在。原始日志保留为PYTEST_BEFORE_BRANCH_ARCHIVE.log。
- 在测试进程中sys.path.insert(0,"src")，不安装包、不改环境文件或tests；移动前：155 passed, 1 warning，7.83s。
- 相同调用移动后：**155 passed, 1 warning，7.20s**；无新增失败，无需恢复文件。warning来自既有test_trainer_base的requires_grad tensor转float。

实际最终命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-vllm-cu128-v011/bin/python -B -c 'import sys; sys.path.insert(0,"src"); import pytest; raise SystemExit(pytest.main(["tests/","-q","-p","no:cacheprovider"]))'
```

## 索引与历史路径

- 仓库根ARCHIVE_INDEX.md为整理后的查找入口；既有SCRIPT_CLASSIFICATION.tsv和方法审计保留为移动前证据，不覆写其历史路径。
- MOVE_LOG.tsv：old_path / new_path / classification / reason / sha256_before / sha256_after。
- COPY_LOG.tsv：113份小报告的原件与副本路径/哈希。
- scripts/archive/_duplicates/INDEX.tsv：15个重复副本与canonical映射。
- OUTPUTS_FUTURE_ARCHIVE.tsv：31个旧outputs路径，只作论文结束后的复核候选，不授权整目录搬迁。所有outputs本轮原位保留。
- 归档源码字节未改，内部parents[2]/相对路径也未改。归档副本用于历史阅读；直接在深层archive运行可能定位错误，当前工作流不依赖这种执行方式。

STOP。
