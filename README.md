# Agentic Model Risk Factory

Discover Agent 的首个可运行版本，围绕 Give Me Some Credit 的 R LightGBM Notebook 建立带证据的模型清单，并提供 Document Agent 的人工审核入口。

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
artifacts/discover/           # 运行生成，仅 inventory.json 提交共享，其余已忽略
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

LangGraph、Streamlit 和后续 Document / Redevelop / Validate 节点由后续模块接入。Discover 的 API 不依赖这些框架。基准预测、R 执行环境和真实数据下载另行准备。

## 测试

```bash
python -m unittest -v
python discover_agent.py schema
```

测试覆盖真实 Notebook、稳定编号、目录分组、Notebook 单元引用、CSV 表头、伪造证据、LLM 重试与失败回退、审批与源文件变动、记录篡改、坏文件、外部符号链接以及输出目录约束。
