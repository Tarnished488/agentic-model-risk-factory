# Agent 系统处理流程详解

## 一、全局流程图

```
                        ┌────────────────────────────────────┐
                        │  输入:legacy_models/ 目录 + 数据集   │
                        └──────────────────┬─────────────────┘
                                           ▼
                                 ① Discover Agent
                                           │ inventory.json
                                           ▼
                                    【人工关卡 1】
                                           ▼
                                 ② Document Agent
                                   ├─ 生成文档
                                   └─ 自检 Agent 交叉核对
                                           │ docs/{id}.md + consistency_report.json
                                           ▼
                                    【人工关卡 2】
                                           ▼
                                 ③ Redevelop Agent
                                   解析 → 中间表示 → 生成代码 → 生成测试
                                           ▼
                                   Parity Check(纯代码)
                                      │FAIL(<3轮)→ 带差异反馈回到 ③
                                      │FAIL(≥3轮)→ 转人工
                                      │PASS
                                           ▼
                                    【人工关卡 3】
                                           ▼
                                 ④ Validate(代码计算 + Explainer Agent)
                                           │ metrics.json + report.pdf
                                           ▼
                                    【人工关卡 4】
                                           ▼
                                 ⑤ Monitor Agent
                                    │ 触发条件命中 → 回到 ④ 重新验证
                                    ▼
                                 alerts.json + 仪表盘

        贯穿全程:audit_log.jsonl(每次调用)+ MLflow(每个节点一个 run)
```

---

## 二、共享状态(所有节点读写)

| 字段                           | 含义                   |
| ------------------------------ | ---------------------- |
| `model_id`                     | 当前处理的模型编号     |
| `inventory_record`             | 该模型的清单记录       |
| `doc_path`                     | 文档路径               |
| `ir_path`                      | 中间表示(逻辑摘要)路径 |
| `generated_code_path`          | 新 Python 模型路径     |
| `parity_result`                | 一致性结果             |
| `retry_count`                  | 重试次数               |
| `metrics_path` / `report_path` | 指标与报告             |
| `approvals`                    | 审批记录列表           |
| `status`                       | 当前阶段状态           |

---

## 三、各节点详细说明

### ① Discover Agent(发现与建清单)

**输入**
- 遗留模型目录(`.sas`、`.R`、`.py`、`.docx`、`.pdf`、`.csv` 数据字典)

**处理过程**
1. **文件扫描(代码)**:递归遍历目录,按扩展名分类,按文件名、目录和代码头部注释把同一模型的代码和文档分组
2. **元数据抽取(LLM)**:对每组文件让 LLM 提取名称、类型(信贷/欺诈/定价)、语言、输入变量、输出、目标变量、负责人、版本、最后修改时间
3. **结构校验(代码)**:用 Pydantic 校验 JSON,格式错误则带报错信息重试,最多 2 次
4. **文档状态判定(代码 + LLM)**:对比文档日期与代码最后修改日期;让 LLM 判断文档内容是否与代码一致,标注 `current / outdated / missing`
5. **风险分级(规则 + LLM)**:按模型类型、用途、影响范围给出初步风险等级,并附理由

**输出**
- `inventory.json`:每个模型一条记录

```json
{
  "model_id": "PD-001",
  "name": "Retail PD Scorecard",
  "type": "credit",
  "language": "SAS",
  "source_files": ["pd_model.sas", "pd_spec_v1.docx"],
  "inputs": ["age", "income", "utilization", "..."],
  "target": "default_flag",
  "doc_status": "outdated",
  "risk_tier": "high",
  "evidence": "代码最后修改 2023-05,文档为 2020 版"
}
```

**人工关卡 1**:用户在表格中核对、修正字段,点"批准"后进入下一步。

---

### ② Document Agent(逆向文档)

**输入**
- `inventory.json` 中被选中的模型记录
- 模型源码、数据字典、旧文档(若有)

**处理过程**
1. **代码切块(代码)**:按数据步骤、变量变换、模型拟合、评分四段切分,避免一次塞入过长代码
2. **分章节生成(LLM)**:每章单独调用,章节对应 SS1/23 要求:
   - 模型目的与适用范围
   - 数据来源与样本定义
   - 变量选择与变换(缺失值处理、截断、分箱、编码)
   - 方法论(算法、参数、评分公式)
   - 假设与已知局限
   - 使用限制与监控建议
3. **证据引用**:要求每条陈述标注来源,如 `[pd_model.sas:L42-58]`
4. **自检 Agent(第二个 LLM 调用)**:输入"生成的文档 + 原代码",逐条核对,输出不一致项,例如"文档称缺失值用中位数填补,代码实际填 0"
5. **合并整理**:不一致项标记在文档中,待人工确认

**输出**
- `docs/PD-001.md`:结构化模型文档
- `consistency_report.json`:自检结果(一致 / 不一致 / 无法确认,每条附证据)

**人工关卡 2**:审阅文档和不一致项,可直接编辑,批准后锁定为"权威文档",作为 Redevelop 的输入。

---

### ③ Redevelop Agent(重构,核心节点)

**输入**
- SAS/R 源码
- 已批准的文档
- 数据字典
- (重试时)上一轮 parity 差异反馈

**处理过程**

**3.1 解析 Agent**:读源码,输出**中间表示(IR)**,即与语言无关的结构化逻辑摘要:
```json
{
  "preprocessing": [
    {"var": "income", "op": "impute", "method": "median"},
    {"var": "utilization", "op": "cap", "upper": 1.5},
    {"var": "age", "op": "bin", "edges": [18, 25, 35, 50, 65]}
  ],
  "model": {"type": "logistic", "solver": "newton", "intercept": true},
  "scoring": {"output": "probability"}
}
```
先产出 IR 再写代码,比一步到位更稳,也便于人工核对。

**3.2 代码生成 Agent**:根据 IR 生成 Python pipeline,实现统一接口 `fit / predict_proba / get_coefficients`。

**3.3 测试生成 Agent**:生成 pytest,包括输入输出形状、概率范围 [0,1]、关键变换(截断、分箱边界)的单元测试。

**3.4 沙箱执行(代码)**:在隔离环境运行 pytest,不通过则把报错回传给代码生成 Agent 修复。

**3.5 Parity Check(纯代码,不用 LLM)**:
- 加载参考输出(系数、测试集预测概率)
- 用新模型在同一数据上预测
- 计算:系数最大绝对差、预测概率最大/平均绝对差、AUC 差、KS 差、Spearman 相关
- 对照**事先定好的阈值**输出 PASS/FAIL

**3.6 路由决策**:

| 结果            | 动作                                                         |
| --------------- | ------------------------------------------------------------ |
| PASS            | 进入人工关卡 3                                               |
| FAIL 且重试 < 3 | 生成差异反馈(如"income 缺失处理可能不同,第 5 号变量预测差异最大"),回到 3.2 |
| FAIL 且重试 ≥ 3 | 转人工,附差异分析报告                                        |

**输出**
- `ir.json`、`new_model.py`、`tests/test_new_model.py`
- `parity_result.json`(含每一轮的历史)
- MLflow:每轮一个 nested run,记录代码、指标、PASS/FAIL

**人工关卡 3**:对比 SAS 与 Python 代码、parity 表,批准后新模型在 Model Registry 登记为 `Staging`。

---

### ④ Validate(独立验证)

分成**确定性计算**和**LLM 解读**两层。

**输入**
- 已批准的 Python 模型
- 数据集(训练 / 测试 / OOT 时间外样本)
- 文档、parity 结果
- 受保护特征列表(性别、年龄段等)

**处理过程**

**4.1 性能计算(代码)**:AUC、Gini、KS、Brier、混淆矩阵、校准曲线、OOT 回测

**4.2 稳定性计算(代码)**:整体 PSI、逐变量 CSI、分时间段指标

**4.3 公平性计算(代码,fairlearn)**:按分组计算通过率、TPR/FPR 差、disparate impact、demographic parity difference

**4.4 概念合理性检查(代码 + LLM)**:系数符号是否符合业务直觉(如逾期次数越多风险越高),由代码判断符号,LLM 负责写评价

**4.5 阈值判定(代码)**:对照预设规则给每项打 绿 / 黄 / 红,如 PSI > 0.25 为红

**4.6 Explainer Agent(LLM)**:
- 输入:`metrics.json` + 判定结果 + 文档
- 任务:撰写各章节解读、列出发现(含严重级别)、给出建议和结论倾向(通过 / 有条件通过 / 不通过)
- 约束:**只能引用输入中已有的数字,不得自行计算**

**4.7 数字校验(代码)**:用正则提取报告中的数字,与 `metrics.json` 比对,不一致则打回重写

**4.8 报告组装(代码)**:模板 + 图表 + LLM 段落 → PDF/Word

**输出**
- `metrics.json`、图表图片
- `validation_report.pdf`(章节对齐 SS1/23)
- `findings.json`(发现项、严重级别、建议)
- MLflow:指标、图表、报告作为 artifact

**人工关卡 4**:验证人员审阅,签署结论,模型状态更新为 `Production` 或 `Rejected`。

---

### ⑤ Monitor Agent(监控,选做)

**输入**
- 已上线模型及其验证基线(`metrics.json`)
- 新进来的时间窗口数据(可模拟或人为加漂移)

**处理过程**
1. **逐窗口计算(代码)**:PSI、各变量 CSI、AUC、通过率变化
2. **与基线对比(代码)**:计算偏离幅度
3. **触发规则(代码)**:`PSI > 0.25` 或 `AUC 下降 > 0.03` 或 公平性差异超限
4. **告警解读(LLM)**:说明哪些变量漂移、可能的业务原因、建议动作
5. **自动动作**:触发则创建"重新验证任务",回到节点 ④,并通知人工

**输出**
- `alerts.json`
- 监控趋势数据,供仪表盘展示
- 重新验证任务(若触发)

---

## 四、横切机制

### 4.1 审计日志
每次 agent 调用写入一行 JSONL:

```json
{"time": "...", "node": "redevelop", "agent": "codegen", "model": "gpt-4o",
 "prompt_version": "v3", "input_hash": "...", "output_path": "...", "tokens": 3120}
```

### 4.2 人工关卡实现
- LangGraph 在关卡节点调用 `interrupt`,状态持久化
- Streamlit 展示待审内容,提供 批准 / 驳回 / 修改后批准
- 审批结果写入 `approvals`,驳回则回到上一节点并携带审阅意见

### 4.3 统一错误处理

| 错误类型           | 处理                                     |
| ------------------ | ---------------------------------------- |
| LLM 输出格式不合法 | Pydantic 校验失败 → 带错误信息重试 ≤2 次 |
| 生成代码报错       | 回传 traceback 让 agent 修复             |
| Parity 失败        | 差异反馈循环 ≤3 次,仍失败转人工          |
| API 超时           | 指数退避重试;使用缓存的历史结果          |

---

## 五、端到端示例:PD-001 一次完整运行

| 步骤 | 输入              | 动作                                     | 输出                            |
| ---- | ----------------- | ---------------------------------------- | ------------------------------- |
| 1    | 5 个遗留文件      | Discover 扫描并抽取                      | 清单含 PD-001,文档状态 outdated |
| 2    | 用户在 UI 点批准  | 关卡 1                                   | 进入 Document                   |
| 3    | pd_model.sas      | 生成文档,自检发现"缺失值填 0 而非中位数" | 文档 + 1 条不一致               |
| 4    | 用户修正并批准    | 关卡 2                                   | 文档锁定                        |
| 5    | 源码 + 文档       | 生成 IR、Python 代码、pytest             | new_model.py                    |
| 6    | 新模型 + 参考输出 | 第 1 轮 parity:预测最大差 0.03 → FAIL    | 反馈"income 截断规则不同"       |
| 7    | 差异反馈          | 第 2 轮修复后 parity:最大差 2e-7 → PASS  | parity_result.json              |
| 8    | 用户批准          | 关卡 3                                   | 登记 Staging                    |
| 9    | 模型 + 数据       | 计算三维度指标,LLM 写解读,数字校验通过   | report.pdf                      |
| 10   | 验证人签署        | 关卡 4                                   | 状态 Production                 |
| 11   | 模拟漂移数据      | PSI 0.31 → 触发告警                      | alerts.json,自动创建重验任务    |

整个过程用时从传统的 8–12 周压缩到几分钟(含人工审阅时间则为小时级),并且每一步都有日志、MLflow 记录和审批痕迹可追溯。

---

## 六、实现优先级提示

| 优先级     | 节点                                 |
| ---------- | ------------------------------------ |
| 必须可演示 | ③ Redevelop + Parity、④ Validate     |
| 应该有     | ① Discover、② Document               |
| 有余力     | ⑤ Monitor、数字校验(4.7)、自检 Agent |

需要的话,我可以接着给你写其中某个节点的**代码骨架**(如 LangGraph 工作流含人工关卡,或 Redevelop 的 parity 反馈循环),或画一张**可视化架构图**用于 PPT。你想要哪个?