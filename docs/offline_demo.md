# 无 API 真实数据演示

这是“真实 R 参考 + 手写 Python 重建 + 独立验证”的演示，未调用大模型，未证明自动转换成功。Document 使用离线档案草稿。所有数据行只保留本地。

## 一次性准备

1. 使用 Python 3.10+ 虚拟环境，安装 requirements.txt。
2. `python -m scripts.download_gmsc` 下载公开第三方镜像，核对已固定的 SHA-256。原竞赛在 https://www.kaggle.com/competitions/GiveMeSomeCredit/data ，官方数据受竞赛条款约束；镜像不改变原始使用条件。不要把镜像称为经认证的官方原件。
3. 安装 R 4.4+，执行 `Rscript scripts/gmsc_probit_export.R`。原 R 脚本所需依赖从 CRAN 安装。R 运行结束后，data/reference/gmsc_probit/ 包含系数、留出预测、训练 ID、行数和 sessionInfo。
4. `python -m scripts.run_offline_demo`。依次执行 Discover、离线 Document、基准重建、Parity、Validate，并导出聚合演示。
5. `python -m http.server 8765 --bind 127.0.0.1 --directory outputs/demo`，打开 http://127.0.0.1:8765 。仅服务脱敏后的演示目录，勿把仓库根目录或 data/ 当服务器根目录。

## 审核如何处理

演示前三个关卡的 reviewer 明确写为“Codex technical review — team human sign-off still required”。这些是 AI 执行的技术检查，不是团队成员的独立人工签名。真实团队需要复核原始源码、离线说明、IR、实现及 Parity，使用自己的姓名和具体意见重新审核。演示不会自动批准关卡 4。规则全部通过也仍等待人工处置未知项。

本次技术核对的范围是来源记录存在、IR 的引用与原 R 源码相符、静态代码检查和单元测试通过、独立执行 R 后预测/系数比较通过。不包含模型业务批准、数据代表性、公平性法律结论或银行准入。

## 产物与交接

- artifacts/offline_demo/latest.json：本机各阶段的审核快照路径和验证状态路径。
- artifacts/offline_demo/validate/runs/<id>/：metrics.json、报告 Markdown / HTML、state.json。
- outputs/demo/：只含聚合指标的 HTML / JSON。scripts.evidence_pack 重新验证第三关审批链与上游数据摘要后导出；不复制原始数据、逐行预测、最差样本或审核人身份。
- docs/evidence/：本次运行的聚合证据及下载摘要，可在 GitHub 审阅。

修改源码、数据或预测后必须从适当阶段重跑，旧的快照不再适用。Validate 在计算前后都复查上游链。关卡 4 审核时再次检查原始链、指标和报告摘要。

## 验证范围

独立重新计算 AUC、Gini、KS、Brier、Log-loss、ECE；给出分箱校准表、训练/留出特征 PSI、示例年龄分组的 TPR/FPR/选择率，并明确最小组及类别样本量。样本不足不判公平性通过。检查原始标签、ID 唯一性、训练留出不重叠、过滤人口覆盖以及有效概率。

IR 描述的过滤规则决定评估人口；过滤本身是否合理仍需要人工判断。分组阈值为研究示例，不能成为实际授信政策。没有时间戳和未来样本，因此 PSI 是切分分布比较，不是上线漂移监控。单一年龄分组也不覆盖完整公平性评估。保留原 R 的预处理顺序，为复现而保留的缺陷应在验证报告中单列，不能用迁移一致性掩盖。

## 当前仍待实现

真实 LLM 自动生成运行（本次按用户要求不配置 API）、LightGBM/XGBoost 全面重建、多模型批量编排、生产模型注册表、持续 Monitor、独立团队签署。静态白名单和独立子进程不等于容器安全隔离。
