# 上传资源分析报告

## 📊 资源概览

您上传了两个关键资源，对我们的Neuro-Symbolic心脏移植决策系统至关重要：

### 资源1：中文医学数据（完整 txt.zip）
- **数量**：7个权威指南文档
- **来源**：中华医学会器官移植学分会
- **版本**：2019版
- **总大小**：~330KB
- **语言**：中文

### 资源2：Graph RAG Agent 开源代码
- **项目**：完整的GraphRAG + DeepSearch实现
- **架构**：Plan-Execute-Report多Agent系统
- **技术栈**：Python + Neo4j + FastAPI + Streamlit
- **核心亮点**：多Agent协同、知识图谱增强检索

---

## 📚 资源1详细分析：中文医学指南

### 文件清单与内容概要

| 文件名 | 大小 | 核心内容 | 可用性 |
|--------|------|----------|--------|
| **consensus_中国心脏移植供心获取与保护技术规范** | 8.7KB | 供心获取流程、冷保存技术、运输方案 | ⭐⭐⭐⭐⭐ |
| **consensus_中国心脏移植术操作规范** | 12KB | 手术步骤、吻合方法（双腔静脉法/双房法/全心法） | ⭐⭐⭐⭐⭐ |
| **consensus_中国心脏移植免疫抑制治疗及排斥反应诊疗规范** | 26KB | 免疫抑制方案、排斥反应诊断和治疗 | ⭐⭐⭐⭐⭐ |
| **consensus_中国心脏移植受者术前评估与准备技术规范** | 25KB | 受者筛选标准、禁忌症、术前准备 | ⭐⭐⭐⭐⭐ |
| **consensus_中国心脏移植术后并发症诊疗规范** | 8.2KB | 常见并发症（出血、感染、PGD）及处理 | ⭐⭐⭐⭐⭐ |
| **consensus_中国心脏移植术后随访技术规范** | 15KB | 随访流程、监测指标、长期管理 | ⭐⭐⭐⭐ |
| **book_心脏外科学ch21** | 222KB | 心脏移植章节完整内容 | ⭐⭐⭐⭐ |

### 数据特点

#### ✅ 优势
1. **权威性**：中华医学会官方规范，直接可作为知识库的"黄金标准"
2. **结构化**：内容分章节，有明确的标题层级
3. **实操性**：详细的手术步骤、吻合方法、参数范围
4. **覆盖全面**：从术前评估 → 手术操作 → 术后管理的完整流程

#### ⚠️ 挑战
1. **中文语言**：需要中文NLP处理，AutoSchemaKG默认是英文
2. **格式简单**：纯文本，缺少结构化标记（JSON/XML）
3. **实体隐式**：风险因子、并发症等实体需要NER抽取
4. **关系隐式**：因果关系、时序关系需要从文本中推理

### 示例内容分析

**来自《中国心脏移植术操作规范》**：

```
## 1 受者病心切除
### 1.1 要点
既往未实施过胸骨劈开术的受者，通常在供心到达前1小时做皮肤切口；
既往实施过心脏手术，则将时间延长至2h，以便有充足时间进行二次开胸
及分离粘连，完全解剖游离受者自身心脏。

### 1.2 操作程序及方法
（1）常规术前准备，消毒，铺巾，取胸正中切口，锯开胸骨。
（2）纵行切开心包，常规探查心脏，充分游离上、下腔静脉和主、肺动脉，
肝素化后准备体外循环。上、下腔静脉插管，位置尽量靠近远心端，
主动脉插管位置靠近无名动脉起始部的升主动脉远端。
（3）上、下腔静脉套上阻断带，开始体外循环并降温至28~32℃，
阻断上、下腔静脉及升主动脉。根据术式确定心脏切除和保留组织范围。
```

**可以抽取的知识**：
- **手术步骤**（Surgical_Step）：
  - "受者病心切除" → "体外循环准备" → "主动脉阻断" → ...
- **时序关系**（TEMPORAL_PRECEDES）：
  - "供心到达前1小时" → "做皮肤切口"
- **条件判断**（Conditional）：
  - IF "既往实施过心脏手术" THEN "时间延长至2h"
- **参数范围**（Parameter）：
  - 降温温度：28~32℃

### 数据用途规划

#### 用途1：作为AutoSchemaKG的输入数据
```
完整 txt/ → AutoSchemaKG → Neo4j
         ↓
      抽取实体+关系
         ↓
   心脏移植知识图谱
```

**优势**：
- 数据已经清洗，直接可用
- 内容覆盖手术全流程，质量高
- 中文内容适合构建中文医学KG

**需要调整**：
- AutoSchemaKG的Prompt需要改为中文（已在`medical_transplant_prompt.py`中预留了中文版）
- Schema需要适配中文实体类型

#### 用途2：作为规则库（Rule Engine）
```
从指南中提取硬性规则：
- 禁忌症（Contraindications）
- 阈值（Thresholds）：如"ischemic_time > 4h"
- 手术步骤（Workflow Templates）
```

**示例规则提取**：

| 规则类型 | 来源段落 | 规则定义 |
|---------|----------|----------|
| **时间阈值** | "供心到达前1小时做皮肤切口" | `cut_start_time = arrival_time - 1h` |
| **温度范围** | "降温至28~32℃" | `28 <= cpb_temperature <= 32` |
| **手术时序** | "先切除心脏后，用电刀分离主动脉和肺动脉" | `heart_removal PRECEDES vessel_separation` |
| **禁忌症** | （免疫规范）"活动性恶性肿瘤" | `IF donor.malignancy.active THEN ABORT` |

#### 用途3：作为框架的Mock KG数据源
在AutoSchemaKG构建真实KG之前，可以：
1. 手工抽取10-20条高质量三元组
2. 作为框架Phase 1的Mock数据
3. 验证框架的RAG Agent能否正确检索和使用

---

## 🤖 资源2详细分析：Graph RAG Agent架构

### 架构概览

这个开源项目与我在Round 1设计的**三Agent系统惊人地相似**！

**对比表**：

| 我的设计（Round 1） | Graph RAG Agent | 一致性 |
|---------------------|----------------|--------|
| **Medical Expert Agent** (Planner & Parser) | **Planner** (Clarifier + TaskDecomposer + PlanReviewer) | ⭐⭐⭐⭐ |
| **RAG Agent** (Graph Builder) | **Executor** (RetrievalExecutor + ResearchExecutor) | ⭐⭐⭐⭐⭐ |
| **Analyzer Agent** (Neuro-Symbolic Reasoner) | **Reporter** (OutlineBuilder + SectionWriter) | ⭐⭐⭐ |
| **LangGraph编排** | **MultiAgentOrchestrator** | ⭐⭐⭐⭐⭐ |

### 核心组件深度分析

#### 1. Planner（规划器）

**与我们的Medical Expert Agent的对应关系**：

```python
# Graph RAG Agent的Planner
Clarifier:        澄清查询 → 识别歧义
TaskDecomposer:   分解任务 → 构建任务依赖图
PlanReviewer:     审核计划 → 生成PlanSpec

# 我们的Medical Expert Agent应该做的
Clinical Parser:  解析临床输入 → 识别关键实体（供体特征、受体特征）
Task Identifier:  识别决策需求 → "需要评估供受体匹配度"
Plan Generator:   生成检索计划 → "查询'缺血时间>4h'的风险数据"
```

**可借鉴的设计模式**：

1. **TaskGraph（任务依赖图）**：
```python
# 他们的设计
class TaskGraph:
    nodes: List[TaskNode]  # 任务节点
    execution_mode: str    # sequential/parallel/adaptive

    def get_ready_tasks() -> List[TaskNode]  # 获取可执行任务
    def topological_sort() -> List[TaskNode]  # 拓扑排序

# 可以应用到我们的手术流程图
class SurgicalWorkflowGraph:
    steps: List[SurgicalStep]  # 手术步骤
    execution_mode: str         # sequential（手术必须按顺序）

    def get_next_step_with_risks() -> Tuple[SurgicalStep, List[RiskFactor]]
```

2. **PlanSpec（计划规范）**：
```python
# 他们的设计
class PlanSpec:
    plan_id: str
    problem_statement: ProblemStatement
    assumptions: List[str]           # 假设前提
    task_graph: TaskGraph
    acceptance_criteria: AcceptanceCriteria  # 验收标准

# 可以应用到我们的决策流程
class TransplantDecisionPlan:
    case_id: str
    clinical_statement: ClinicalCase  # 供受体信息
    assumptions: List[str]            # 如"供心质量良好"
    workflow_graph: SurgicalWorkflowGraph
    decision_criteria: DecisionCriteria  # 如"PGD风险<10%"
```

#### 2. Executor（执行器）

**与我们的RAG Agent的对应关系**：

```python
# Graph RAG Agent的Executor
RetrievalExecutor:   执行检索任务 → 调用Neo4j查询
ResearchExecutor:    深度研究 → 多步推理
ReflectionExecutor:  反思结果 → 质量评估

# 我们的RAG Agent应该做的
KG Retriever:        检索相似病例 → 从Neo4j查询
Risk Assessor:       风险评估 → 基于检索结果计算风险
Graph Builder:       构建手术流程图 → 动态生成Patient-Specific Graph
```

**可借鉴的核心工具**：

1. **EvidenceTracker（证据追踪器）**：
```python
# 他们的设计
class EvidenceTracker:
    def add_evidence(result: RetrievalResult):
        """按来源与粒度去重，优先保留高分证据"""

    def get_evidence_by_source(source: str) -> List[RetrievalResult]:
        """按来源查询证据"""

# 可以应用到我们的系统
class ClinicalEvidenceTracker:
    def add_case_evidence(case: SimilarCase, relevance_score: float):
        """记录相似病例，按相关度排序"""

    def get_cases_by_risk_factor(risk_factor: str) -> List[SimilarCase]:
        """查询包含特定风险因子的病例"""

    def get_contradicting_evidence(claim: str) -> List[Evidence]:
        """查询矛盾的证据（用于Veto协议）"""
```

2. **RetrievalResult（检索结果）**：
```python
# 他们的设计
class RetrievalResult:
    result_id: str
    granularity: str  # "DO/L2-DO/Chunk/AtomicKnowledge/KGSubgraph"
    evidence: Union[str, Dict[str, Any]]
    source: str       # 检索来源
    score: float      # 相关度分数（0.0-1.0）

# 可以应用到我们的系统
class MedicalRetrievalResult:
    case_id: str
    granularity: str  # "SimilarCase/RiskFactor/SurgicalStep/Guideline"
    evidence: Union[CaseData, RiskData, GuidelineText]
    source: str       # "Neo4j_KG" / "Rule_Engine" / "Literature"
    confidence: float # 证据强度（RCT=0.9, Cohort=0.7, Case=0.5）
```

#### 3. Reporter（报告生成器）

**与我们的Analyzer Agent的对应关系**：

```python
# Graph RAG Agent的Reporter
OutlineBuilder:     生成报告大纲 → 结构化输出
SectionWriter:      章节写作 → 基于证据生成文本
ConsistencyChecker: 一致性检查 → 验证逻辑

# 我们的Analyzer Agent应该做的
Risk Aggregator:    风险聚合 → 从GNN+LLM获取风险评分
Decision Synthesizer: 决策综合 → 融合双路推理结果
Explanation Generator: 解释生成 → 生成决策报告（含可解释性）
```

**可借鉴的Map-Reduce模式**：

```python
# 他们用于处理大量证据的策略
EvidenceMapper:   将证据分批映射为摘要
SectionReducer:   将摘要归约为连贯文本

# 可以应用到我们的风险评估
RiskMapper:       将多个相似病例映射为风险特征
RiskReducer:      将所有风险特征归约为最终风险评分
```

### WorkerCoordinator（工作协调器）- 重点！

这是最值得借鉴的组件，负责任务调度：

```python
class WorkerCoordinator:
    def __init__(self, execution_mode: str = "sequential"):
        """
        execution_mode:
        - sequential: 串行执行，严格按依赖顺序
        - parallel: 并行执行，使用线程池并发执行独立任务
        """

    def execute_plan(self, plan_signal: PlanExecutionSignal):
        """
        根据execution_mode调度任务：
        - sequential: for循环顺序执行
        - parallel: ThreadPoolExecutor并发执行无依赖任务
        """
```

**对我们的启示**：

在手术决策系统中，某些操作可以并行：
```python
# 并行检索
parallel_tasks = [
    retrieve_similar_cases(donor_features),      # 任务1
    retrieve_risk_factors(ischemic_time),        # 任务2
    retrieve_guidelines(recipient_condition)     # 任务3
]
results = ThreadPoolExecutor().map(execute, parallel_tasks)

# 串行推理（必须按顺序）
sequential_tasks = [
    build_workflow_graph(results),               # 必须等检索完成
    run_gnn_inference(workflow_graph),           # 必须等图构建完成
    generate_explanation(gnn_results)            # 必须等推理完成
]
for task in sequential_tasks:
    task.execute()
```

---

## 🔗 资源整合策略

### 整合方案：三层架构

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: 知识层 (Knowledge Layer)                        │
│  ┌────────────────────────────────────────────────────┐  │
│  │  中文医学指南 → AutoSchemaKG → Neo4j KG            │  │
│  │  (完整 txt.zip)     (医学Prompt)   (心脏移植KG)    │  │
│  └────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 2: Agent层 (借鉴Graph RAG Agent架构)              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Medical      │  │ RAG Agent    │  │ Analyzer     │  │
│  │ Expert       │─→│ (Executor模式)│─→│ (Reporter模式)│  │
│  │ (Planner模式)│  │              │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│         │                   │                  │        │
│  ┌──────▼───────────────────▼──────────────────▼─────┐  │
│  │    MultiAgentOrchestrator (WorkerCoordinator)    │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│  Layer 3: 推理层 (Neuro-Symbolic Layer)                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │  GNN (GraphSAGE) + LLM Reasoning                  │  │
│  │  - GNN读取手术流程图                               │  │
│  │  - LLM结合KG检索结果推理                           │  │
│  │  - 决策融合 (Veto协议)                            │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 实施路径

#### Phase 1: KG构建（2-3天）
1. 使用AutoSchemaKG处理7个中文指南
2. 修改Prompt为中文医学版本
3. 生成初版Neo4j KG
4. 手工验证100个三元组质量

#### Phase 2: Agent框架搭建（3-5天）
1. 参考Graph RAG Agent的架构
2. 实现三个核心Agent：
   - Medical Expert (借鉴Planner的TaskDecomposer)
   - RAG Agent (借鉴Executor的RetrievalExecutor + EvidenceTracker)
   - Analyzer (借鉴Reporter的结构化输出)
3. 实现MultiAgentOrchestrator编排逻辑

#### Phase 3: Neuro-Symbolic推理集成（5-7天）
1. 实现GNN模型（GraphSAGE）
2. 实现手术流程图动态构建
3. 实现决策融合和Veto协议
4. 端到端测试

---

## 🎯 立即可执行的任务

### 任务1: 处理中文医学数据
**目标**：用AutoSchemaKG从7个指南中构建KG

**步骤**：
```bash
cd /home/user/SHU/AutoSchemaKG

# 1. 准备数据（已完成）
# 数据在：/home/user/SHU/data_txt/完整 txt/

# 2. 修改配置使用中文Prompt
python example/medical_transplant_kg_extraction.py \
  --data_dir /home/user/SHU/data_txt/完整\ txt/ \
  --language zh-CN \
  --output_dir ./chinese_medical_kg

# 3. 检查输出
cat chinese_medical_kg/kg_extraction/*.json | jq '.'
```

**注意事项**：
- AutoSchemaKG默认英文，需要使用`medical_transplant_prompt.py`中的中文版本
- 中文NER可能需要额外配置（ScispaCy不支持中文，考虑用pkuseg或jieba）

### 任务2: 分析Graph RAG Agent代码
**目标**：提取可复用的组件到我们的框架

**重点文件**：
```
graph_agent/graph-rag-agent-master/graphrag_agent/agents/multi_agent/
├── orchestrator.py          # 编排逻辑
├── planner/task_decomposer.py  # 任务分解
├── executor/worker_coordinator.py  # 任务调度
├── tools/evidence_tracker.py  # 证据追踪
└── core/                    # 数据模型
    ├── plan_spec.py
    ├── execution_record.py
    └── state.py
```

**提取清单**：
- [ ] MultiAgentOrchestrator的编排模式
- [ ] TaskGraph的依赖管理逻辑
- [ ] EvidenceTracker的去重和排序算法
- [ ] WorkerCoordinator的并行执行策略
- [ ] PlanSpec / ExecutionRecord的数据模型

### 任务3: 设计Knowledge Access Layer (KAL)
**目标**：定义KG查询接口，确保Agent和KG解耦

**接口定义**（基于Graph RAG Agent的RetrievalAdapter）：

```python
class KnowledgeAccessLayer:
    """
    统一的知识图谱访问接口
    """

    def query_similar_cases(
        self,
        patient_features: Dict[str, Any],
        top_k: int = 5
    ) -> List[SimilarCase]:
        """
        检索相似病例

        Args:
            patient_features: 供受体特征
            top_k: 返回最相似的k个病例

        Returns:
            List[SimilarCase]: 相似病例列表，按相似度排序
        """
        pass

    def query_surgical_workflow_template(
        self,
        criteria: Dict[str, Any]
    ) -> SurgicalWorkflowTemplate:
        """
        查询手术流程模板

        Args:
            criteria: 查询条件（如供体类型、受体风险等级）

        Returns:
            SurgicalWorkflowTemplate: 手术流程模板
        """
        pass

    def query_risk_factors(
        self,
        context: Dict[str, Any]
    ) -> List[RiskFactor]:
        """
        查询风险因子

        Args:
            context: 临床上下文（如缺血时间、供体年龄）

        Returns:
            List[RiskFactor]: 风险因子列表，含权重
        """
        pass

    def query_contraindications(
        self,
        patient_profile: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        查询禁忌症

        Args:
            patient_profile: 供受体档案

        Returns:
            {"absolute": [...], "relative": [...]}
        """
        pass
```

---

## 📝 后续工作优先级

| 优先级 | 任务 | 依赖 | 预计时间 |
|--------|------|------|----------|
| 🔴 **P0** | 用AutoSchemaKG处理7个中文指南 | 无 | 1天 |
| 🔴 **P0** | 手工验证KG质量，抽查100个三元组 | P0-1 | 0.5天 |
| 🟠 **P1** | 提取Graph RAG Agent的可复用组件 | 无 | 2天 |
| 🟠 **P1** | 实现Knowledge Access Layer接口 | P0-1 | 1天 |
| 🟡 **P2** | 实现Medical Expert Agent（参考Planner） | P1 | 3天 |
| 🟡 **P2** | 实现RAG Agent（参考Executor） | P0-2, P1-2 | 3天 |
| 🟢 **P3** | 实现Analyzer Agent（GNN+LLM融合） | P2 | 5天 |
| 🟢 **P3** | 集成MultiAgentOrchestrator | P2 | 2天 |

---

## 💡 关键发现与建议

### 发现1: Graph RAG Agent架构完美契合我们的需求
**证据**：
- 他们的Planner-Executor-Reporter与我们的Medical Expert-RAG-Analyzer一一对应
- MultiAgentOrchestrator的编排逻辑可以直接复用
- EvidenceTracker的证据管理正是我们需要的"相似病例映射"功能

**建议**：
- 不要重新发明轮子，直接基于他们的代码框架改造
- 重点改造：
  - Planner → 适配医学决策场景
  - Executor → 添加GNN推理路径
  - Reporter → 改为决策报告生成

### 发现2: 中文医学数据质量极高，但需要中文NLP
**证据**：
- 7个指南都是权威官方文档
- 内容详细、结构清晰
- 覆盖手术全流程

**建议**：
- AutoSchemaKG的Prompt已预留中文版本，但需要测试
- 中文实体识别可能需要替换模型：
  - 选项1: 使用中文医学预训练模型（如MC-BERT）
  - 选项2: 使用通用中文LLM（如Qwen, GLM）
  - 选项3: 翻译成英文后处理（不推荐，丢失语义）

### 发现3: 可以快速搭建MVP
**证据**：
- 数据已就绪（7个指南）
- 代码框架已有参考（Graph RAG Agent）
- 架构设计已完成（Round 1）

**建议MVP范围**：
1. **Week 1**: KG构建（AutoSchemaKG + 中文指南）
2. **Week 2**: 三Agent框架（复用Graph RAG Agent架构）
3. **Week 3**: Neuro-Symbolic推理（GNN + 决策融合）
4. **Week 4**: 端到端测试和优化

---

## 🎯 下一步行动

**立即可做（今天）**：
1. 测试AutoSchemaKG处理中文数据
2. 提取Graph RAG Agent的核心组件到我们的项目
3. 设计Knowledge Access Layer的详细接口

**等待您确认**：
1. 是否先用小规模测试（1个指南）验证中文处理？
2. 是否需要我立即开始实现KAL接口？
3. 您希望优先看到哪个部分的代码实现？

**请告诉我您的偏好，我会立即开始工作！** 🚀
