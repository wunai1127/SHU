# 可集成组件与系统整合策略

## 一、Graph RAG Agent可复用组件

基于对`graph-rag-agent-master`代码的分析，以下组件可直接集成到心脏移植AI系统：

### 1.1 核心可复用组件

#### ✅ **EvidenceTracker** (证据追踪器)
**位置**: `graphrag_agent/agents/multi_agent/evidence_tracker.py`

**功能**:
- 自动去重检索结果
- 按相关性评分排序
- 追踪证据来源

**集成价值**:
- 在Medical Expert Agent中，追踪每个临床决策的证据来源
- 支持"为什么推荐这个手术方案"的可解释性

**集成方式**:
```python
# 集成到Medical Expert Agent
from graphrag_agent.agents.multi_agent.evidence_tracker import EvidenceTracker

class MedicalExpertAgent:
    def __init__(self):
        self.evidence_tracker = EvidenceTracker(
            similarity_threshold=0.85,  # 医学证据要求更高相似度
            max_evidence_per_source=5
        )

    def retrieve_clinical_guidelines(self, query: str) -> List[Evidence]:
        """检索临床指南并追踪证据"""
        raw_results = self.neo4j_retriever.search(query)

        # 去重并评分
        deduplicated = self.evidence_tracker.add_and_deduplicate(raw_results)

        return deduplicated
```

---

#### ✅ **TaskGraph** (任务图协调器)
**位置**: `graphrag_agent/agents/multi_agent/task_graph.py`

**功能**:
- 表示任务依赖关系（DAG）
- 支持串行/并行执行
- 任务状态追踪

**集成价值**:
- 管理手术流程步骤的依赖关系（如"体外循环建立" → "主动脉阻断"）
- 支持动态调整手术流程

**集成方式**:
```python
from graphrag_agent.agents.multi_agent.task_graph import TaskGraph, Task

class SurgicalWorkflowManager:
    def __init__(self):
        self.task_graph = TaskGraph()

    def build_surgical_workflow(self, patient_case: Dict) -> TaskGraph:
        """根据患者情况构建手术流程图"""
        # 定义手术步骤
        tasks = [
            Task(id="prepare_cpb", name="建立体外循环", dependencies=[]),
            Task(id="clamp_aorta", name="主动脉阻断", dependencies=["prepare_cpb"]),
            Task(id="excise_heart", name="切除病心", dependencies=["clamp_aorta"]),
            Task(id="implant_donor", name="植入供心", dependencies=["excise_heart"])
        ]

        # 根据患者风险因子动态添加任务
        if patient_case['pulmonary_hypertension']:
            tasks.append(Task(
                id="pvr_management",
                name="肺血管阻力管理",
                dependencies=["prepare_cpb"],
                priority="high"
            ))

        self.task_graph.add_tasks(tasks)
        return self.task_graph
```

---

#### ✅ **WorkerCoordinator** (多Agent协调器)
**位置**: `graphrag_agent/agents/multi_agent/orchestrator.py`

**功能**:
- 协调多个Agent按顺序或并行执行
- 异常处理与重试
- 结果聚合

**集成价值**:
- 直接作为MultiAgentOrchestrator的基础框架
- 管理Medical Expert Agent、RAG Agent、Analyzer Agent的协作

**集成方式**:
```python
from graphrag_agent.agents.multi_agent.orchestrator import WorkerCoordinator

class HeartTransplantOrchestrator(WorkerCoordinator):
    def __init__(self):
        super().__init__()
        self.medical_expert = MedicalExpertAgent()
        self.rag_agent = RAGAgent()
        self.analyzer_agent = AnalyzerAgent()

    async def process_case(self, patient_data: Dict) -> Decision:
        """处理单个移植案例"""
        # Step 1: Medical Expert解析输入
        clinical_features = await self.execute_worker(
            self.medical_expert,
            "parse_patient_data",
            patient_data
        )

        # Step 2: RAG Agent检索相似案例
        similar_cases = await self.execute_worker(
            self.rag_agent,
            "retrieve_similar_cases",
            clinical_features
        )

        # Step 3: Analyzer Agent推理
        decision = await self.execute_worker(
            self.analyzer_agent,
            "analyze_and_decide",
            {
                "patient": clinical_features,
                "similar_cases": similar_cases
            }
        )

        return decision
```

---

#### ✅ **GraphRetriever** (图检索器)
**位置**: `graphrag_agent/graph/retriever.py`

**功能**:
- 多跳子图检索
- 路径查询（A → B的所有路径）
- 邻居扩展

**集成价值**:
- RAG Agent的核心组件
- 支持"延长缺血时间 → ? → 原发性移植物功能障碍"的推理路径发现

**集成方式**:
```python
from graphrag_agent.graph.retriever import GraphRetriever

class RAGAgent:
    def __init__(self, neo4j_config: Dict):
        self.retriever = GraphRetriever(
            neo4j_uri=neo4j_config['uri'],
            neo4j_user=neo4j_config['username'],
            neo4j_password=neo4j_config['password']
        )

    def find_risk_pathways(self, risk_factor: str, complication: str) -> List[Path]:
        """查找风险因子到并发症的推理路径"""
        query = f"""
        MATCH path = (r:Entity {{name: '{risk_factor}'}})-[*1..3]->(c:Entity {{name: '{complication}'}})
        RETURN path, length(path) as hops
        ORDER BY hops
        LIMIT 10
        """
        paths = self.retriever.execute_cypher(query)
        return paths

    def retrieve_similar_cases(self, patient_features: Dict) -> List[CaseGraph]:
        """检索相似病例的子图"""
        # 1. 找到相似的风险因子组合
        risk_factors = patient_features['risk_factors']
        query = f"""
        MATCH (p:Patient)-[:HAS_RISK]->(r:RiskFactor)
        WHERE r.name IN {risk_factors}
        WITH p, count(r) as overlap
        WHERE overlap >= {len(risk_factors) * 0.7}
        MATCH path = (p)-[*1..2]-(related)
        RETURN path
        LIMIT 50
        """
        subgraphs = self.retriever.execute_cypher(query)
        return subgraphs
```

---

### 1.2 部分可改造组件

#### 🔧 **Planner组件**（需适配医学场景）
**位置**: `graphrag_agent/agents/multi_agent/planner.py`

**原功能**:
- 任务分解（Clarifier → TaskDecomposer → PlanReviewer）
- 适用于通用问答任务

**改造方向**:
```python
class MedicalCasePlanner(BasePlanner):
    """医学案例规划器（改造自原Planner）"""

    def decompose_medical_task(self, case: Dict) -> List[Task]:
        """将医学决策任务分解为子任务"""
        tasks = []

        # 子任务1: 供体评估
        if 'donor_features' in case:
            tasks.append(Task(
                type="donor_evaluation",
                description="评估供体质量和适配性",
                input=case['donor_features']
            ))

        # 子任务2: 受体风险评估
        tasks.append(Task(
            type="recipient_risk_assessment",
            description="计算受体风险评分",
            input=case['recipient_features']
        ))

        # 子任务3: 手术方案选择
        tasks.append(Task(
            type="surgical_plan_selection",
            description="选择最佳手术方案",
            dependencies=["donor_evaluation", "recipient_risk_assessment"]
        ))

        return tasks
```

---

#### 🔧 **Reporter组件**（改造为决策解释器）
**位置**: `graphrag_agent/agents/multi_agent/reporter.py`

**原功能**:
- 生成结构化研究报告（Outline → Section Writing → Consistency Check）

**改造方向**:
```python
class ClinicalDecisionReporter(BaseReporter):
    """临床决策解释器"""

    def generate_decision_report(self, decision: Decision) -> Report:
        """生成可解释的决策报告"""
        report = Report()

        # Section 1: 决策摘要
        report.add_section("决策摘要", self._format_decision(decision))

        # Section 2: 证据支持
        report.add_section("证据支持", self._format_evidence(decision.evidence))

        # Section 3: 风险量化
        report.add_section("风险量化", self._format_risk_scores(decision.risks))

        # Section 4: 替代方案
        report.add_section("替代方案", self._format_alternatives(decision.alternatives))

        return report
```

---

## 二、AutoSchemaKG与Graph RAG Agent的集成架构

### 2.1 三层集成架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Layer 3: Neuro-Symbolic Agent              │
│  ┌────────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │Medical Expert  │  │  RAG Agent    │  │ Analyzer Agent  │ │
│  │(改造Planner)   │  │(GraphRetriever)│  │ (GNN+LLM)      │ │
│  └────────┬───────┘  └───────┬───────┘  └────────┬────────┘ │
│           └──────────────┬───────────────────────┘          │
│                          │                                   │
│                  ┌───────▼────────┐                          │
│                  │WorkerCoordinator│                         │
│                  │ (Orchestrator)  │                         │
│                  └───────┬─────────┘                         │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│                    Layer 2: Knowledge Access Layer            │
│                  ┌───────▼─────────┐                          │
│                  │ EvidenceTracker │                          │
│                  │  TaskGraph      │                          │
│                  └───────┬─────────┘                          │
│                          │                                    │
│         ┌────────────────┼────────────────┐                  │
│         ▼                ▼                ▼                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │Neo4j API │    │PyG Graph │    │Cache API │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────┼───────────────────────────────────┐
│                    Layer 1: Knowledge Storage                 │
│                  ┌───────▼─────────┐                          │
│                  │   Neo4j DB      │                          │
│                  │ (心脏移植KG)     │                          │
│                  └─────────────────┘                          │
│                          ▲                                    │
│                          │                                    │
│                  ┌───────┴─────────┐                          │
│                  │ AutoSchemaKG    │                          │
│                  │ (KG构建流水线)   │                          │
│                  └─────────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 数据流向

```
输入: 20000+医学文献 (JSON)
  │
  ▼
[AutoSchemaKG Pipeline]  ← 使用 chinese_medical_kg_schema.json
  │
  ├─ LLM抽取三元组
  ├─ Schema验证
  ├─ 实体归一化
  │
  ▼
[Neo4j Database]
  │ (导入100万+三元组)
  │
  ▼
[Knowledge Access Layer]
  │ ← Graph RAG Agent的GraphRetriever
  │ ← EvidenceTracker去重
  │
  ▼
[Multi-Agent System]
  │
  ├─ Medical Expert: 解析患者数据
  ├─ RAG Agent: 检索相似案例子图
  ├─ Analyzer Agent: GNN+LLM推理
  │
  ▼
输出: 移植决策 + 风险评分 + 可解释性报告
```

---

## 三、具体集成代码示例

### 3.1 统一配置文件

```yaml
# integrated_system_config.yaml

# 知识图谱构建（AutoSchemaKG）
kg_construction:
  schema_file: "/home/user/SHU/schemas/chinese_medical_kg_schema.json"
  llm:
    provider: "deepseek"
    api_key: "YOUR_API_KEY"
  neo4j:
    uri: "bolt://localhost:7687"
    username: "neo4j"
    password: "YOUR_PASSWORD"
    database: "heart_transplant_kg"

# Multi-Agent系统（Graph RAG Agent改造）
multi_agent:
  orchestrator:
    type: "WorkerCoordinator"
    max_parallel_workers: 3

  agents:
    medical_expert:
      type: "Planner"  # 复用Graph RAG Agent的Planner
      base_class: "graphrag_agent.agents.multi_agent.planner.Planner"
      customization: "MedicalCasePlanner"

    rag_agent:
      type: "GraphRetriever"  # 直接使用GraphRetriever
      base_class: "graphrag_agent.graph.retriever.GraphRetriever"
      neo4j: "kg_construction.neo4j"  # 引用上面的Neo4j配置

    analyzer_agent:
      type: "NeuoSymbolicReasoner"  # 自定义新组件
      gnn_model_path: "/home/user/SHU/models/gnn_risk_predictor.pt"
      llm_model: "deepseek-chat"

  utilities:
    evidence_tracker:
      enabled: true
      similarity_threshold: 0.85
      base_class: "graphrag_agent.agents.multi_agent.evidence_tracker.EvidenceTracker"

    task_graph:
      enabled: true
      base_class: "graphrag_agent.agents.multi_agent.task_graph.TaskGraph"

# GNN配置
gnn:
  model_type: "GraphSAGE"
  hidden_dim: 128
  num_layers: 3
  training:
    epochs: 100
    learning_rate: 0.001
    batch_size: 32
```

### 3.2 集成启动脚本

```python
# integrated_system/main.py

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "AutoSchemaKG"))
sys.path.append(str(Path(__file__).parent.parent / "graph_agent/graph-rag-agent-master"))

from automated_kg_pipeline.auto_kg_builder import AutoKGBuilder
from graphrag_agent.agents.multi_agent.orchestrator import WorkerCoordinator
from graphrag_agent.agents.multi_agent.evidence_tracker import EvidenceTracker
from graphrag_agent.graph.retriever import GraphRetriever

class IntegratedHeartTransplantSystem:
    """集成的心脏移植AI决策系统"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        # 初始化知识图谱（如果未构建）
        if not self._check_kg_exists():
            print("知识图谱未构建，开始自动构建...")
            self._build_kg()

        # 初始化Multi-Agent系统
        self._init_agents()

    def _build_kg(self):
        """构建知识图谱（调用AutoSchemaKG流水线）"""
        kg_builder = AutoKGBuilder(self.config['kg_construction'])
        kg_builder.run()
        print("✓ 知识图谱构建完成")

    def _check_kg_exists(self) -> bool:
        """检查Neo4j中是否已有KG"""
        neo4j_config = self.config['kg_construction']['neo4j']
        retriever = GraphRetriever(
            neo4j_uri=neo4j_config['uri'],
            neo4j_user=neo4j_config['username'],
            neo4j_password=neo4j_config['password']
        )
        result = retriever.execute_cypher("MATCH (n) RETURN count(n) as cnt")
        return result[0]['cnt'] > 0

    def _init_agents(self):
        """初始化Multi-Agent系统"""
        # 1. 初始化工具组件
        self.evidence_tracker = EvidenceTracker(
            similarity_threshold=self.config['multi_agent']['utilities']['evidence_tracker']['similarity_threshold']
        )

        self.graph_retriever = GraphRetriever(
            neo4j_uri=self.config['kg_construction']['neo4j']['uri'],
            neo4j_user=self.config['kg_construction']['neo4j']['username'],
            neo4j_password=self.config['kg_construction']['neo4j']['password']
        )

        # 2. 初始化Agent
        self.medical_expert = MedicalExpertAgent()
        self.rag_agent = RAGAgent(self.graph_retriever, self.evidence_tracker)
        self.analyzer_agent = AnalyzerAgent(
            gnn_model_path=self.config['multi_agent']['agents']['analyzer_agent']['gnn_model_path']
        )

        # 3. 初始化协调器
        self.orchestrator = WorkerCoordinator()
        self.orchestrator.register_worker("medical_expert", self.medical_expert)
        self.orchestrator.register_worker("rag_agent", self.rag_agent)
        self.orchestrator.register_worker("analyzer_agent", self.analyzer_agent)

        print("✓ Multi-Agent系统初始化完成")

    async def process_transplant_case(self, patient_data: Dict, donor_data: Dict) -> Decision:
        """处理单个移植案例（端到端流程）"""
        # Step 1: Medical Expert解析输入
        clinical_context = await self.orchestrator.execute_worker(
            "medical_expert",
            "parse_case",
            {"patient": patient_data, "donor": donor_data}
        )

        # Step 2: RAG Agent检索相似案例
        similar_cases = await self.orchestrator.execute_worker(
            "rag_agent",
            "retrieve_similar_cases",
            clinical_context
        )

        # Step 3: Analyzer Agent推理决策
        decision = await self.orchestrator.execute_worker(
            "analyzer_agent",
            "analyze_and_decide",
            {
                "context": clinical_context,
                "similar_cases": similar_cases
            }
        )

        return decision


# 使用示例
if __name__ == '__main__':
    # 初始化系统
    system = IntegratedHeartTransplantSystem('integrated_system_config.yaml')

    # 处理案例
    patient = {
        "age": 55,
        "pvr": 4.5,  # Wood单位
        "lvef": 15,  # %
        "creatinine": 1.8  # mg/dL
    }

    donor = {
        "age": 35,
        "ischemic_time": 4.5,  # 小时
        "left_ventricular_hypertrophy": False
    }

    decision = await system.process_transplant_case(patient, donor)
    print(decision)
```

---

## 四、集成优先级

### 阶段1: 基础集成（本周）
✅ **立即可用**：
1. GraphRetriever → RAG Agent
2. EvidenceTracker → 证据追踪
3. WorkerCoordinator → Agent协调

### 阶段2: 适配改造（下周）
🔧 **需适配**：
1. Planner → MedicalCasePlanner
2. Reporter → ClinicalDecisionReporter
3. TaskGraph → SurgicalWorkflowManager

### 阶段3: 深度定制（第三周）
🆕 **需新建**：
1. AnalyzerAgent（GNN+LLM）
2. 物理约束的风险量化模块
3. Veto协议决策融合模块

---

## 五、集成后的系统能力

| 能力 | 来源 | 说明 |
|------|------|------|
| **知识图谱构建** | AutoSchemaKG | 从20000篇文献自动构建KG |
| **图检索** | Graph RAG Agent | 多跳推理、路径查询 |
| **证据去重** | Graph RAG Agent | 避免重复证据影响决策 |
| **任务协调** | Graph RAG Agent | Multi-Agent流程管理 |
| **GNN推理** | 自定义 | 图结构风险量化 |
| **LLM推理** | 自定义 | 语义理解与解释 |
| **决策融合** | 自定义 | Veto协议 |

---

## 六、下一步行动

### 立即可执行（今天）：
1. ✅ 将Graph RAG Agent代码库复制到项目目录
2. ✅ 安装依赖：`pip install -r graph_agent/requirements.txt`
3. ✅ 测试GraphRetriever连接Neo4j

### 本周任务：
1. 构建知识图谱（使用AutoSchemaKG + 您的20000篇文章）
2. 集成GraphRetriever到RAGAgent原型
3. 实现MedicalExpertAgent的基础解析逻辑

### 下周任务：
1. 实现AnalyzerAgent（GNN模块）
2. 集成WorkerCoordinator
3. 端到端测试单个案例
