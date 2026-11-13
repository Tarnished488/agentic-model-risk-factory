# Discover Agent 到 Document Agent 的交接约定

Document Agent 接收一条人工批准过的模型记录及其来源材料，用于编写模型说明书。数据在磁盘上保存为 UTF-8 JSON，进入 Python 时为 dict。当前交接入口是 discover_review.load_document_input，代码已经实现；本页后半部分的 Document 输出格式属于后续节点的实现约定。

## 一分钟理解

Discover 交付三样东西。

1. **模型档案**，模型叫什么、用什么语言、预测什么、提取结果来自哪里、哪些信息仍未知。
2. **源码和配套说明**，包括 Notebook 代码单元、原有文档、字段说明及数据来源。提取出的文本已保存在档案中，原文件用于验证版本和后续深入分析。
3. **人工审核记录**，谁允许这条档案进入文档编制、何时审核、依据哪个版本、保留了哪些问题。

Document 根据这些材料编写说明，遇到缺失信息继续列为待确认。允许编写文档与模型通过验证属于不同阶段。

## 交接文件与入口

| 文件或材料 | 格式 | 用途 |
| --- | --- | --- |
| artifacts/discover/inventory.json | JSON 对象，含 models 数组 | 最新扫描清单，供检查和选择模型 |
| artifacts/discover/reviews/<run_id>/<model_id>/<review_id>.json | JSON 对象 | 实际交接的审核快照，含 approval、inventory_record、input_root、status |
| legacy_models/<group>/ 下的原文件 | R、Notebook、数据字典、说明等 | loader 对照哈希和提取文本，确认资料未变 |
| schemas/inventory.schema.json | JSON Schema | 扫描清单的机器校验定义，Record 与 Metadata 也在其中定义 |

**接收方使用审核快照调用 loader。** inventory.json 本身没有审批状态，不能作为通过人工关卡的凭据。reviews 是本地运行生成目录，已被 .gitignore 排除，不会自动出现在 GitHub；artifacts 下只有最新的 inventory.json 提交共享。

```python
from pathlib import Path
from discover_review import load_document_input

review_file = Path(
    "artifacts/discover/reviews/实际运行号/实际模型号/实际审核号.json"
)
state = load_document_input(review_file)
record = state["inventory_record"]
metadata = record["metadata"]
sources = record["sources"]
```

路径使用 review 命令实际返回的值。loader 校验批准状态、记录摘要、源文件哈希、提取文本和引用。校验失败时先解决原因，再重新交接。

## Python 返回值

| 字段 | 类型 | 接收方处理 |
| --- | --- | --- |
| model_id | str | 文档和报告沿用同一编号 |
| inventory_record | dict | 模型记录，结构见下一节 |
| input_root | str | 输入根目录；在仓库内扫描时为相对仓库根目录的路径（如 legacy_models），需在仓库根目录运行，否则为绝对路径 |
| status | str，当前固定为 ready_for_document | 标识 loader 检查通过，可以开始编制 |
| approvals | list[dict] | 原样保留人工审核信息 |

一个返回值对应一个模型。批量处理时，对每个模型各自选定的批准记录调用 loader。这个返回值可作为 LangGraph 共享状态的一部分，由 D 显式合并。当前 loader 不会自动调用 Document，也不设置 doc_path。

注意三层状态不同：审核文件的 status 为 approved_for_document；loader 返回的 status 为 ready_for_document；inventory_record.status 保留扫描时的 pending_review 快照。判断入口可用性以 loader 成功返回为准。

## 模型记录包含什么

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| model_id、group | str | 固定模型号及相对分组目录 |
| source_files | list[dict] | 原文件相对路径、SHA-256、字节数和文件修改时间 |
| sources | list[dict] | 可引用的文本片段、来源标识、语言提示和文件哈希 |
| metadata | dict | 提取出的业务与技术信息 |
| risk_tier | str 或 null | high、medium、low 或尚未分级 |
| risk_reason | str | 初步分级依据或待确认原因 |
| risk_provisional | bool，固定为 true | 提示等级仍需人工确认 |
| extraction_mode | str | offline 或 llm |
| status | str | 扫描状态，待审核或需处理错误 |
| warnings | list[str] | 扫描与提取问题 |

source_files 中的 modified_at 是文件系统信息，不能直接充当模型发布日期或文档失效证据。

### metadata 字段类型

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| name、algorithm、dataset、target、output | str 或 null | 名称、算法、数据来源、目标和输出说明 |
| business_type | credit、fraud、pricing、other 或 null | 业务类别 |
| language | R、Python、SAS、mixed 或 null | 代码语言 |
| inputs、derived_features | list[str] | 原始输入与派生特征分别记录 |
| data_references | list[str] | 源码中的数据路径引用，不保证该路径在本机存在 |
| training_stages、packages | list[str] | 同一工作流的训练阶段及依赖 |
| business_owner、declared_version | str 或 null | 业务管理人及声明版本；作者与管理人分开 |
| use_in_decisions | direct_credit_decision、reporting、research 或 null | 材料明确说明的用途 |
| doc_status | current、outdated、missing、needs_review | Discover 的初步文档状态 |
| doc_reason | str | 文档状态理由 |
| evidence | dict[str, list[dict]] | 字段名称映射到引用列表 |
| unknowns | list[str] | 待核实的问题 |

空值约定统一采用 JSON null、Python None；列表未确定时使用 []，并在 unknowns 说明缺口。Document 应把缺口写入限制或待确认章节。没有完整输入列表时，不能把几个观察到的变量写成全部变量。

## 源码引用怎样传递

每个 sources 元素有 source_id、path、kind、sha256、text、language_hint。

- path 相对于 input_root。
- source_id 在一次模型扫描中标识具体片段。
- Notebook 标识使用 original.ipynb#cell=3 这类形式，单元序号从 1 开始，Markdown 单元也计数。
- line_start、line_end 从 1 开始，针对该片段 text 的 splitlines() 结果。
- PDF 片段按页编号；DOCX 的行号针对提取文本。CSV 只有表头。

以下是当前样例能够使用的一条真实引用形状，展示 library 调用的位置。

```json
{
  "source_id": "gmsc_lightgbm/original.ipynb#cell=3",
  "line_start": 3,
  "line_end": 3,
  "quote": "library(lightgbm)"
}
```

Document 可以把它显示为：

```text
[gmsc_lightgbm/original.ipynb#cell=3:L3]
```

编写文档时应读取 sources 的全文，再定位 evidence。Discover 的摘要和引用是分析线索，仍需核对语义。原作者的竞赛分数与描述应明确归属原作者，不能写成本项目已复现结果。

如需定位原文件，可以这样读取已校验的清单路径。

```python
input_root = Path(state["input_root"])
for item in record["source_files"]:
    original_path = input_root / item["path"]
```

## 当前 LightGBM 示例的具体交接内容

- 一个业务模型工作流，包含两次 LightGBM 训练。
- 语言 R，目标变量 SeriousDlqin2yrs，数据引用为 Give Me Some Credit。
- 原 Notebook 的代码和 Markdown 说明，来源与版本记录，以及配套材料缺口说明。
- 离线模式已识别目标、依赖和两次训练调用。
- 原始数据及完整字段说明尚未加入，因此原始输入清单、目标的准确业务解释与预测期限仍待核验。
- 业务管理人、真实使用范围和正式风险等级仍待确认。
- 训练好的模型文件、R 基准预测、独立验证结果应由后续任务提供。

开始写代码逻辑、训练阶段和依赖说明所需材料已经具备。数据定义和业务适用性章节应保留明确的缺口。

## 不同成员电脑之间怎样交接

推荐 A、B、C、D 使用同一个 GitHub 分支版本。在仓库根目录运行时 input_root 为相对路径，同一分支的成员可在各自电脑上校验；审核快照仍依赖本地原文件，原文件哈希须一致。

第一轮联调建议由 D 在同一台机器运行 scan 和人工 review，然后把返回的审核文件路径交给 Document。其他成员可在自己的克隆目录重新运行并审阅本地结果，形成新的本地审批。

如果后续要跨机器延续同一份审批，由 D 实现明确的目录重定位流程，保留原审核记录，验证每份文件哈希后再调用。当前 CLI 尚未提供重定位参数。手动改动模型记录或源码，会触发摘要或版本校验。

## 建议 Document 节点输出

以下是接下来实现 Document 时建议采用的接口，当前 Discover 尚未生成这些文件。

| 输出 | 类型或格式 | 内容 |
| --- | --- | --- |
| docs/<model_id>.md | UTF-8 Markdown | 用途、数据、变量变换、方法、假设、限制和待确认项，技术与事实描述附引用 |
| consistency_report.json | JSON 对象 | model_id、源 record_sha256、检查项数组和总体待确认项 |
| 共享状态中的 doc_path | str | 新文档路径 |
| 共享状态中的 status | str | 例如 pending_document_review，由团队统一命名 |

检查项建议包含 claim_id、statement、result、reason、evidence；result 使用 consistent、inconsistent、unverifiable。将每项证据继续使用上述 source_id 和行号格式。

文档和一致性报告绑定 approvals 中的 record_sha256 与 run_id，避免不同扫描或人工修改版本混用。文档编制完成后进入人工关卡 2，获得批准后再交给 Redevelop。

## 接收方联调清单

1. 能成功导入并调用 load_document_input。
2. 能读出模型编号、metadata、sources 和 approvals。
3. 能从原 Notebook 第 3 单元找到 library(lightgbm)，生成带位置引用的说明。
4. 输入缺失时，文档保留待确认项。
5. 源码变化或审批记录不匹配时，停止编制并返回错误。
6. 输出模型文档和一致性报告，并关联原运行号与审核摘要。
