# Agentic Model Risk Factory

当前实现 Discover → 人工关卡 1 → Document 逆向文档与自检 → 人工关卡 2 → Redevelop 输入交接，围绕 Give Me Some Credit 的 R LightGBM Notebook 提供示例。后续模型重构与验证节点尚未实现。

## 先运行离线示例

需要 Python 3.10 或以上。在项目目录运行。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python discover_agent.py scan
python -m unittest -v
```

Windows 激活命令为 `.venv\Scripts\activate`。离线模式无需 API 密钥，不执行 R、不训练模型。示例已包含下载的完整 Notebook 代码和来源记录，移除了历史输出和执行计数。它是 Apache 2.0 授权的第三方示例，归原作者 brynning 所有，详见 `THIRD_PARTY_NOTICES.md`。

预期识别一个业务模型工作流、R、LightGBM、`SeriousDlqin2yrs` 和两次训练调用。原始数据尚未加入，离线扫描的 `inputs` 保持空列表并列为待确认事项。算法识别等静态规则主要针对本次 R 样例，语义更复杂的源码使用 LLM 模式并人工核对。

## 输入和分组约定

```text
legacy_models/
  gmsc_lightgbm/
    original.ipynb
    source.json
    DATA_README.md
artifacts/discover/           # 运行生成，已忽略
  inventory.json             # 最新结果
  runs/<run_id>/inventory.json
  audit_log.jsonl
  reviews/<run_id>/<model_id>/<review_id>.json
```

输入根目录的每个直接子目录代表一个业务模型，内部代码和材料递归归组。根目录的散落文件作为一个 `_root` 组，因此共用说明请放在输入目录以外。单个子目录里有多个业务模型时，先拆分目录，避免仅凭相似文件名自动合并。模型编号由相对分组路径生成，重跑保持稳定，改目录名会改变编号。

支持 `.R`、`.Rmd`、`.py`、`.sas`、`.ipynb`、`.md`、`.txt`、`.json`、`.csv`、`.docx`、`.pdf`。Notebook 仅读代码和 Markdown 单元，引用采用原单元序号和单元内行号。PDF 按页提取文本，扫描件须先 OCR。DOCX 的表格转为文本行。CSV 仅读表头，文件哈希仍会流式计算全文件。官方 Excel 数据字典应先另存为 CSV 后加入。其他扩展名跳过。

文本文件和文档上限 10 MB，CSV 不受该限制。LLM 输入上限 120,000 个字符，超限会显式转人工，不静默截断。源码符号链接不可指向输入目录之外。

## LLM 元数据提取

支持提供 `/chat/completions` 和 JSON 输出的兼容服务。根据实际服务填写以下环境变量，密钥保存在本地环境中。

```bash
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_MODEL="your-model-name"
export LLM_API_KEY="your-api-key"
python discover_agent.py scan --llm
```

显式使用 `--llm` 后，提取的源码、文档文字和 CSV 表头会发送至配置的服务。Notebook 历史输出和 CSV 数据行不会发出。依赖用 urllib，校验 TLS 证书，HTTP 错误不输出响应正文或密钥。

LLM 负责名称、用途、语言、算法、原始输入与派生特征、目标、输出、训练阶段、包、管理人、声明版本、文档状态及未知项。模型源码和资料被当作待分析内容，提示词约束其忽略材料内的指令。

Pydantic 检查字段和取值，代码核对证据来源、行号和逐字引用。引用能证明出处存在，语义是否支持结论仍需人工检查。初次提取失败最多重试两次，总共三次。失败后保留离线结果，明确标为 `needs_attention`，命令返回非零；不会将回退结果伪装为 LLM 提取成功。

每个字段的证据集中在 `metadata.evidence`，例子为 `original.ipynb#cell=3` 及该单元内的行号。模型记录还保存完整提取文本和原文件 SHA-256，方便下一节点读取与重验。

## 文档和风险规则

有可读说明时默认为 `needs_review`，没有说明则为 `missing`。LLM 可以提出 `current` 或 `outdated`，必须引用内容证据，仍属于待审核结果。文件系统修改时间作为文件信息保留，不用于宣称文档已经过时。

业务管理人与 Notebook 作者分别处理。源码没有的管理人、业务范围和模型声明版本保留为空，并在未知项中说明。

只有资料明确说明用途时才应用项目初步分级规则：直接信贷决策为 high，报表为 medium，研究为 low。用途未知则不定级。所有等级都标为 provisional，供人工确认；这套简化规则不构成完整的银行风险分类政策或监管认证。

## 人工审核和下一节点交接

负责 Document 的同学请先阅读 [Discover 到 Document 的交接约定](docs/discover_to_document.md)，其中包含字段类型、证据引用、跨机器路径限制及建议输出。

运行 `scan` 后，打开 `artifacts/discover/inventory.json`，检查来源、目标、原始字段与派生变量、文档状态和待确认事项。下面的编号对应当前样例，新增模型后使用命令输出的实际编号。

```bash
python discover_agent.py review \
  --model-id MODEL-D05BB473A58E \
  --reviewer "Your name" \
  --decision approve \
  --notes "核对过代码证据；业务用途和输入字段仍待后续补充，允许进入文档编制"
```

也可以用 `--decision reject` 并填写原因。这里的批准仅表示允许继续编制文档，不表示模型通过独立验证。CLI 审核记录由操作者填写姓名，身份认证和界面签署由后续平台实现。

每次审核生成独立的 review 文件，保存审核人、意见、时间、记录摘要和完整模型记录。复制该命令返回的文件路径完成交接。

```bash
python discover_agent.py handoff artifacts/discover/reviews/RUN_ID/MODEL_ID/REVIEW_ID.json
```

Document Agent 可以直接调用。

```python
from pathlib import Path
from discover_agent import load_document_input

state = load_document_input(Path("实际审核文件路径.json"))
# state 包含 model_id、inventory_record、input_root、approvals 和 status
```

交接时重新检查批准状态、记录摘要、原文件哈希、提取文本与证据；源文件或批准后的记录改动会被阻止。每次新扫描有独立 run_id，旧批准不会自动应用到新结果。

如需纠正识别结果，推荐修改或补充输入材料后重扫。也可在审核前编辑最新清单的 metadata，保留有效来源证据，并在审核意见中写清修改内容。每次扫描的原始记录保存在 runs 下，审核另存快照。`needs_attention` 的扫描必须先解决问题并重跑才能批准。

## 审计与 MLflow

默认记录本地 JSONL，包括运行号、模型编号、提取模式、调用重试和结果、输入摘要、审核动作。并发写入同一输出目录会被锁阻止。程序意外退出留锁时，确认该目录没有运行中的任务后删除 `.discover.lock` 再重试。不同任务应使用不同输出目录。

可选接入 MLflow，每次扫描一个 run，保存模型数量、异常数量和清单。跟踪服务失败时本地清单仍保留，命令返回非零。

```bash
python -m pip install mlflow-skinny
export MLFLOW_TRACKING_URI="file:./mlruns"
python discover_agent.py scan --mlflow
```

## 当前完成范围

已实现 Discover 扫描、离线提取、可配置 LLM 提取、Pydantic 与证据校验、初步文档和风险状态、CLI 人工审核、版本校验及 Document 交接接口。尚未配置真实 LLM 服务或 MLflow 服务，在线调用通过模拟响应测试，后续需要联调。

Document 及 CLI 人工关卡 2 见下文。LangGraph、Streamlit 和后续 Redevelop / Validate 节点尚未接入。Discover 的 API 不依赖这些框架。基准预测、R 执行环境和真实数据下载另行准备。

## 测试

```bash
python -m unittest -v
python discover_agent.py schema
```

测试覆盖真实 Notebook、稳定编号、目录分组、Notebook 单元引用、CSV 表头、伪造证据、LLM 重试与失败回退、审批与源文件变动、记录篡改、坏文件、外部符号链接以及输出目录约束。

## Document Agent：源码逆向文档

`document_agent.py` 接收 Discover `review` 命令返回的批准快照路径。

```bash
source .venv/bin/activate
# 离线草稿，转录已有清单，不执行语义自检
python document_agent.py generate DISCOVER_REVIEW.json
# 在线生成，使用 LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
python document_agent.py generate DISCOVER_REVIEW.json --llm
# 可选：每个 Document 节点一次 MLflow run（先安装 mlflow-skinny）
python document_agent.py generate DISCOVER_REVIEW.json --llm --mlflow
```

旧形式 `python document_agent.py DISCOVER_REVIEW.json --llm` 保持兼容。

源码按数据、变量变换、拟合、评分阶段启发式切块，并受字符预算约束；说明、字典及未分类代码也完整保留。切块是上下文窗口，不是完整的 SAS/R 语法解析器，跨块依赖必须保留为未知项。引用始终使用原 source_id 和原局部行号。单行过长或完整请求超限会显式失败，不截断。

相邻源窗口完整打包到约 40000 字符的材料组；当前示例的全部材料仅一组，因此每章只调用一次 LLM，而不逐小窗口重复生成。每章每组最多提炼 12 条重要事实，未知项和建议单列；相同陈述合并引用。自检把陈述分批，与所有材料组交叉核对；有一处矛盾时优先保留 inconsistent，所有组均无支持时为 unverifiable。LLM 自检不保证语义正确，仍需人工审核。事实必须附原文引用，建议单独呈现。

每次调用初次尝试加最多两次重试，携带结构或引用错误反馈，间隔指数退避。默认单次运行最多 30 次实际 API 尝试（含重试，缓存命中不计）；`--max-calls` 可显式调整。达到预算会保留草稿并标记 needs_attention。CLI 在请求前后打印阶段、调用计数和结果，运行目录的 progress.json 同步记录进度。Ctrl+C 中断会保存部分结果并释放锁。默认启用经校验的本地响应缓存，键包含完整提示词、材料、JSON Schema、服务地址和模型；`--no-cache` 强制重新调用。`--chunk-characters` 可调整默认 12000 字符的源窗口预算。显式 `--llm` 会向配置服务发送提取材料。

输出位于 `artifacts/document/runs/<run_id>/`：

| 文件 | 内容 |
| --- | --- |
| docs/<model_id>.md | 六章文档、引用及自检结论 |
| chunks.json | 原始行号、阶段提示和完整源窗口 |
| sections.json | 结构化章节，逐步保存以保留已完成内容 |
| check_windows.json | 各材料组的核对结论，逐批保存 |
| progress.json | 当前阶段、API 尝试数、预算和完成状态 |
| consistency_report.json | 陈述、结论、理由、证据及版本关联 |
| state.json | 共享状态、审批历史、路径、摘要、重试计数及失败原因 |

生成前后均重验 Discover 批准快照和源码。成功为 `pending_document_review`；发现矛盾或生成/自检失败为 `needs_attention`。失败保留已完成章节，模式标记 `partial_llm`，命令返回非零。离线模式为 `offline`，自检项均为 `unverifiable`。

### 人工关卡 2 和 Redevelop 交接

打开 `state.json` 中的 doc_path 和 consistency_report_path 进行审核。可直接编辑草稿 Markdown；编辑稿、自检未确认项或离线稿的批准必须填写 resolution-notes，记录人工核对及处理结果。技术失败的运行必须重跑，不能直接批准。

```bash
python document_agent.py review DOCUMENT_STATE.json \
  --reviewer "Your name" --decision approve --notes "已核对文档与源码"
# 修改稿或存在待确认项时
python document_agent.py review DOCUMENT_STATE.json \
  --reviewer "Your name" --decision approve --notes "完成审核" \
  --resolution-notes "说明具体修改，以及矛盾和未知项的人工处置"
# 驳回，随后将意见传入新一轮生成
python document_agent.py review DOCUMENT_STATE.json \
  --reviewer "Your name" --decision reject --notes "请修正缺失值处理描述"
python document_agent.py generate DISCOVER_REVIEW.json --llm --feedback-review REJECTED_REVIEW.json
# 使用 review 命令输出的快照路径交接
python document_agent.py handoff DOCUMENT_REVIEW.json
```

审核生成独立快照，保存当时的文档全文、报告、审批、原始摘要和编辑后摘要。后续修改草稿不影响已批准版本；修改批准副本或原始源码会阻止交接。摘要锁定属于本地完整性检查，不是数字签名或操作系统不可变存储。CLI 审核人由操作者填写，尚无身份认证。

```python
from pathlib import Path
from document_review import load_redevelop_input
state = load_redevelop_input(Path("DOCUMENT_REVIEW.json"))
# status == ready_for_redevelop；含源码记录、批准文档和两个人工关卡的审批
```

调用审计记录阶段、提示词版本、模型、输入摘要、结果路径、尝试次数及服务返回的 token 用量；未返回用量或缓存命中时记录 null，不估算。MLflow 失败保留本地输出，并返回非零。

共享键在 `workflow_state.py` 定义，后续 ir_path、generated_code_path、parity_result、metrics_path、report_path 初始为空。本版本没有执行模型训练或生成 Parity/Validate 的成功状态。

### 流程对照和验证

完整差异与实现边界见 [流程对照记录](docs/flow_alignment.md)。当前审核通过 CLI 执行；LangGraph interrupt、Streamlit 表格审核、Redevelop、Validate 和 Monitor 尚未实现。真实 LLM / MLflow 服务仍需配置后联调。

```bash
python -m unittest -v
```

测试覆盖静态扫描、源窗口完整性与行号、模拟 LLM 生成和跨窗口自检、重试反馈与退避、缓存、编辑审批、驳回反馈、快照及源码变更阻断、MLflow 失败保留本地结果。

### Document 耗时与调用预算

早期 document-v2 将每章与每个小窗口分别生成，再让所有陈述批次与所有窗口逐一核对；29 个窗口可能产生上百次生成和数百次自检。document-v3 保留源码切块用于定位，但按材料组调用，避免这类乘法膨胀。短模型约为六次章节生成加若干次批量自检，实际次数受输出长度、重试、缓存影响；默认总预算 30 次。

如仍有旧进程运行，先在其终端 Ctrl+C，再使用新版本运行同一 generate 命令。修改源文件不会改变已启动进程的逻辑。旧运行的 sections.json 和 cache 保留，但新提示词/分组与旧缓存不同，不能假定新版本无成本续跑。
