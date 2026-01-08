# 🚨 AutoSchemaKG医学应用：核心建议与注意事项

## 🎯 立即可以做的事（今天）

### 1. 测试AutoSchemaKG的基础功能
```bash
cd /home/user/SHU/AutoSchemaKG

# 准备10篇文章的小测试集
# 使用默认prompt测试pipeline是否work
python example/example_scripts/custom_extraction/custom_kg_extraction.py
```

**目的**：验证AutoSchemaKG能正常运行，LLM配置正确

### 2. 检查您的Schema文件
**您提到"已经有schema"，请回答**：
- Schema是JSON格式还是Python类？
- 是否定义了实体类型枚举？
- 是否定义了关系类型枚举？
- 是否包含医学领域特定的约束？

**请分享**：Schema文件的路径或内容

### 3. 检查您的20000+文章JSON
**请确认格式**：
```bash
# 查看第一篇文章
head -1 /path/to/your/articles.json | jq '.'
```

**必需字段**：
```json
{
  "id": "...",      // 必需
  "text": "...",    // 必需（abstract内容）
  "metadata": {     // 必需
    "lang": "en"    // 必需（AutoSchemaKG用于选择prompt）
  }
}
```

---

## ⚠️ 关键技术决策（需要您确认）

### 决策1: LLM选择

| 选项 | 优势 | 劣势 | 成本估算 |
|------|------|------|---------|
| **OpenAI API** (GPT-4) | 质量最高 | API成本高 | ~$200-400 (20000篇) |
| **本地Llama 3.1 70B** | 零成本 | 需要高端GPU (A100) | 硬件成本 |
| **本地Llama 3.1 8B** | 零成本+低硬件需求 | 质量较低 | 需人工校验更多 |
| **混合方案** | 平衡成本和质量 | 复杂度高 | ~$50-100 |

**我的推荐**：
1. **Phase 1测试**：用Llama 3.1 8B (本地)
2. **Phase 2验证**：抽样100篇用GPT-4验证质量差距
3. **Phase 3生产**：如果差距<10% → 全用Llama 8B；否则用GPT-4

### 决策2: 是否需要UMLS归一化？

**UMLS的作用**：
- 将"Left Ventricular Ejection Fraction"、"LVEF"、"LV EF"映射到同一个CUI
- 支持跨文献的实体对齐

**Trade-off**：
| | 不用UMLS | 用UMLS |
|---|---------|--------|
| **开发复杂度** | 低 | 高（需下载UMLS，配置QuickUMLS） |
| **KG质量** | 中等（重复实体多） | 高（实体标准化） |
| **检索准确性** | 中等 | 高 |

**我的推荐**：
- **Phase 1**：不用UMLS，用AutoSchemaKG自带的concept_generation
- **Phase 2**：如果发现大量重复实体（如"PGD"和"Primary Graft Dysfunction"同时存在），再添加UMLS

### 决策3: AutoSchemaKG的三阶段抽取 vs 单阶段

**AutoSchemaKG默认**：
1. entity_relation（实体-关系）
2. event_entity（事件-实体）
3. event_relation（事件-事件关系）

**医学文献特点**：
- 心脏移植文献主要是**因果关系**（风险因子→并发症）
- 较少描述**事件序列**（event_relation）

**我的推荐**：
```python
# 只用两个阶段
MEDICAL_TRANSPLANT_SCHEMA = {
    "entity_relation": {...},      # 保留
    "surgical_workflow": {...},     # 新增（替代event_relation）
    # "event_entity": {...}         # 删除（医学文献中不常见）
}
```

---

## 🛠️ 必须修改的代码位置

### 修改1: Prompt文件（已为您创建）
**位置**：`/home/user/SHU/AutoSchemaKG/atlas_rag/llm_generator/prompt/medical_transplant_prompt.py`

**已包含**：
- 医学实体类型枚举（Donor, Recipient, Risk_Factor...）
- 医学关系类型（INCREASES_RISK_OF, MITIGATES...）
- 统计量抽取指令（OR, p-value, CI）
- 缩写消歧规则（PGD = Primary Graft Dysfunction）

**需要您做**：
- 根据您的domain knowledge添加更多示例
- 如果您的文献是中文，添加中文版本

### 修改2: Schema文件（已为您创建）
**位置**：`/home/user/SHU/AutoSchemaKG/atlas_rag/llm_generator/format/medical_kg_schema.py`

**已包含**：
- 实体类型枚举
- 关系类型枚举
- Statistical_Metrics字段（支持OR, p-value）
- Evidence_Strength字段（RCT, Cohort...）

**需要您做**：
- 对比您现有的schema，看是否需要添加/删除实体类型

### 修改3: 配置文件（已为您创建）
**位置**：`/home/user/SHU/AutoSchemaKG/example/medical_transplant_kg_extraction.py`

**需要您修改**：
```python
config = ProcessingConfig(
    model_path="your-model-here",  # 改成您的LLM
    data_directory="/path/to/your/20000/articles",  # 改成实际路径
    filename_pattern="your-pattern",  # 改成文件前缀
    ...
)
```

---

## 📋 执行步骤（优先级排序）

### 🔴 高优先级（本周完成）

1. **验证数据格式**
   ```bash
   cd /home/user/SHU
   # 请告诉我：您的20000篇文章存放在哪里？
   # 格式是 *.json 还是 *.jsonl 还是 *.json.gz？
   ```

2. **小规模测试**
   - 用10篇文章测试默认AutoSchemaKG
   - 用10篇文章测试医学特化版本
   - 对比质量差异

3. **人工验证**
   - 抽查50个三元组
   - 计算准确率
   - 识别常见错误类型

### 🟡 中优先级（下周完成）

4. **迭代优化Prompt**
   - 根据验证结果添加few-shot examples
   - 调整实体类型定义

5. **配置并行处理**
   - 20000篇分成20个shard
   - 测试单个shard的处理时间
   - 估算总时间

6. **设计Neo4j Schema**
   - 定义节点标签
   - 定义关系类型
   - 创建索引和约束

### 🟢 低优先级（可选）

7. **添加UMLS归一化**（如果Phase 1发现重复实体严重）

8. **训练自定义NER模型**（如果LLM抽取质量不够）

9. **构建质量监控Dashboard**

---

## 🚧 常见坑（提前避免）

### 坑1: LLM输出被截断
**症状**：某些文章的三元组数量异常少

**原因**：`max_new_tokens`设置太小，LLM输出被截断

**解决**：
```python
config = ProcessingConfig(
    max_new_tokens=2048,  # 默认512可能不够
    ...
)
```

### 坑2: GPU内存溢出
**症状**：`CUDA out of memory`

**解决**：
```python
config = ProcessingConfig(
    batch_size_triple=4,  # 降低批量大小（默认16）
    use_8bit=True,        # 启用8bit量化
    ...
)
```

### 坑3: Neo4j导入超时
**症状**：导入100万三元组卡住

**解决**：
```bash
# 不要用LOAD CSV，用neo4j-admin import
neo4j-admin database import full \
    --nodes=triple_nodes.csv \
    --relationships=triple_edges.csv
```

### 坑4: 实体重复（"PGD" vs "Primary Graft Dysfunction"）
**症状**：同一概念有多个节点

**临时解决**（Phase 1）：
```cypher
// 手工合并
MATCH (a:Entity {name: "PGD"})
MATCH (b:Entity {name: "Primary Graft Dysfunction"})
CALL apoc.refactor.mergeNodes([a,b], {properties: "combine"})
```

**长期解决**（Phase 2）：
- 添加UMLS归一化
- 或在Prompt中强制使用全称

---

## 📞 需要您反馈的信息

请提供以下信息，我将生成定制化的配置：

1. **数据位置**：
   - 20000+文章JSON的路径
   - 文件格式（.json / .jsonl / .json.gz）
   - 一篇示例文章

2. **Schema**：
   - 您现有schema的文件路径
   - 或schema的JSON内容

3. **计算资源**：
   - GPU型号和数量
   - 是否有多机并行能力
   - 是否有API预算

4. **LLM选择**：
   - 本地模型还是API？
   - 如果本地，模型名称
   - 如果API，预算范围

5. **Neo4j**：
   - 本地部署还是云端（Neo4j Aura）？
   - 账号密码（用于生成配置）

---

## ✅ 成功的标志

**Phase 1成功**（测试阶段）：
- [ ] 100篇文章成功抽取三元组
- [ ] 人工验证准确率>80%
- [ ] 能导入Neo4j并查询

**Phase 2成功**（全量阶段）：
- [ ] 20000篇全部处理完成
- [ ] 生成>100万三元组
- [ ] Neo4j查询延迟<2秒

**Phase 3成功**（集成阶段）：
- [ ] KG能回答"相似病例查询"
- [ ] KG能支持手术流程图构建
- [ ] 框架的RAG Agent能调用KG

---

## 🎯 下一条消息请回复

1. 您的20000+文章JSON的**路径**或**样例**
2. 您的Schema文件的**内容**
3. 您希望先做**小规模测试**还是直接**全量处理**？

我将基于您的回复生成：
- 定制化的配置文件
- 一键运行脚本
- 质量验证工具

---

## 📚 参考资源

**AutoSchemaKG文档**：
- GitHub: https://github.com/HKUST-KnowComp/AutoSchemaKG
- Paper: https://arxiv.org/abs/2505.23628

**医学NLP工具**：
- ScispaCy: https://allenai.github.io/scispacy/
- QuickUMLS: https://github.com/Georgetown-IR-Lab/QuickUMLS
- UMLS下载: https://www.nlm.nih.gov/research/umls/

**我创建的文件**：
- 医学Prompt: `/home/user/SHU/AutoSchemaKG/atlas_rag/llm_generator/prompt/medical_transplant_prompt.py`
- 医学Schema: `/home/user/SHU/AutoSchemaKG/atlas_rag/llm_generator/format/medical_kg_schema.py`
- 使用示例: `/home/user/SHU/AutoSchemaKG/example/medical_transplant_kg_extraction.py`
- 实施指南: `/home/user/SHU/MEDICAL_KG_IMPLEMENTATION_GUIDE.md`
