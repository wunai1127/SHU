# 心脏移植医学知识图谱构建实施指南

## 📋 实施路线图

### Phase 1: 数据准备与验证（1-2天）

#### 1.1 检查您的文章JSON格式
```bash
# 查看一篇示例文章的结构
head -1 your_articles.json | jq '.'
```

**必需字段**：
- `id`: 文章唯一标识符
- `text`: Abstract文本
- `metadata`: 元数据（至少包含 `lang: "en"`）

**示例格式**：
```json
{
  "id": "pmid_12345678",
  "text": "Extended donor ischemic time (>4h) was associated with increased risk of primary graft dysfunction (OR=2.1, p<0.001)...",
  "metadata": {
    "lang": "en",
    "title": "Impact of Ischemic Time on Heart Transplant Outcomes",
    "journal": "Circulation",
    "year": 2023,
    "study_type": "Cohort"
  }
}
```

#### 1.2 准备小规模测试集
```bash
# 从20000篇中抽取100篇用于测试
head -100 your_articles.json > test_100.json
```

### Phase 2: 配置AutoSchemaKG（半天）

#### 2.1 修改配置文件
编辑 `example/medical_transplant_kg_extraction.py`:

```python
config = ProcessingConfig(
    model_path="your-llm-model",  # 使用您的LLM
    data_directory="/path/to/test_100.json",  # 先用小数据集测试
    filename_pattern="test",
    output_directory="./test_output",

    # 关键：使用医学特化的prompt和schema
    triple_extraction_prompt_path="./atlas_rag/llm_generator/prompt/medical_transplant_prompt.py",
    triple_extraction_schema_path="./atlas_rag/llm_generator/format/medical_kg_schema.py",

    batch_size_triple=4,  # 小批量测试
    debug_mode=True  # 启用调试
)
```

#### 2.2 测试运行
```bash
cd /home/user/SHU/AutoSchemaKG
python example/medical_transplant_kg_extraction.py
```

**预期输出**：
- `test_output/kg_extraction/`: JSON格式的抽取结果
- `test_output/triples_csv/`: CSV格式的三元组

#### 2.3 验证输出质量
手工检查前20个三元组：

```bash
head -20 test_output/triples_csv/triple_edges_test_from_json_without_emb.csv
```

**检查项**：
- ✅ 实体类型是否正确（Donor, Recipient, Risk_Factor...）
- ✅ 关系类型是否合理（INCREASES_RISK_OF, MITIGATES...）
- ✅ 是否提取了统计量（OR, p-value）
- ✅ 医学缩写是否正确展开（PGD→Primary Graft Dysfunction）

### Phase 3: 医学术语归一化（1-2天）

#### 3.1 安装医学NLP工具
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz
pip install quickumls
```

#### 3.2 创建归一化脚本
创建文件：`post_processing/medical_normalization.py`

```python
import spacy
import pandas as pd
from quickumls import QuickUMLS

# 加载医学NLP模型
nlp = spacy.load("en_core_sci_lg")

# 初始化UMLS链接器（需要下载UMLS数据）
matcher = QuickUMLS("/path/to/umls/data")

def normalize_entity(entity_text):
    """
    将实体映射到UMLS CUI
    """
    matches = matcher.match(entity_text, best_match=True)
    if matches:
        cui = matches[0][0]['cui']
        preferred_term = matches[0][0]['preferred']
        return cui, preferred_term
    return None, entity_text

# 处理CSV文件
df = pd.read_csv("test_output/triples_csv/triple_nodes_test_from_json_without_emb.csv")

for idx, row in df.iterrows():
    cui, normalized = normalize_entity(row['name'])
    df.at[idx, 'umls_cui'] = cui
    df.at[idx, 'normalized_name'] = normalized

df.to_csv("test_output/triples_csv/triple_nodes_normalized.csv", index=False)
```

### Phase 4: Neo4j导入（1天）

#### 4.1 准备Neo4j环境
```bash
# 使用Docker运行Neo4j
docker run \
    --name heart-transplant-kg \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your-password \
    -v $PWD/neo4j_data:/data \
    neo4j:latest
```

#### 4.2 创建Schema约束
在Neo4j Browser中执行：

```cypher
// 创建唯一性约束
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT donor_id IF NOT EXISTS FOR (d:Donor) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT recipient_id IF NOT EXISTS FOR (r:Recipient) REQUIRE r.id IS UNIQUE;

// 创建索引
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX umls_cui IF NOT EXISTS FOR (e:Entity) ON (e.umls_cui);
```

#### 4.3 批量导入数据
```cypher
// 导入节点
LOAD CSV WITH HEADERS FROM 'file:///triple_nodes_normalized.csv' AS row
CREATE (e:Entity {
    id: row.id,
    name: row.name,
    normalized_name: row.normalized_name,
    umls_cui: row.umls_cui,
    type: row.type
})

// 为特定类型添加额外标签
MATCH (e:Entity) WHERE e.type = 'Donor'
SET e:Donor

// 导入关系
LOAD CSV WITH HEADERS FROM 'file:///triple_edges_test_from_json_without_emb.csv' AS row
MATCH (head:Entity {id: row.head})
MATCH (tail:Entity {id: row.tail})
CREATE (head)-[r:RELATION {
    type: row.relation,
    odds_ratio: toFloat(row.odds_ratio),
    p_value: toFloat(row.p_value),
    evidence_strength: row.evidence_strength,
    source: row.source
}]->(tail)
```

### Phase 5: 质量验证与迭代（持续）

#### 5.1 统计分析
```cypher
// 检查实体类型分布
MATCH (e:Entity)
RETURN e.type, count(*) as count
ORDER BY count DESC

// 检查关系类型分布
MATCH ()-[r:RELATION]->()
RETURN r.type, count(*) as count
ORDER BY count DESC

// 查找孤立节点
MATCH (e:Entity)
WHERE NOT (e)--()
RETURN count(e)
```

#### 5.2 医学验证
抽查高影响三元组：

```cypher
// 查找高风险关系（OR > 2.0）
MATCH (rf:Risk_Factor)-[r:RELATION]->(c:Complication)
WHERE r.odds_ratio > 2.0
RETURN rf.name, r.odds_ratio, c.name, r.evidence_strength
ORDER BY r.odds_ratio DESC
LIMIT 20
```

**人工审查清单**：
- [ ] 实体识别是否准确？
- [ ] 关系方向是否正确？
- [ ] 统计量是否合理？
- [ ] 是否有明显的矛盾（如同一风险因子在不同文献中的OR差异巨大）？

#### 5.3 迭代改进
根据验证结果调整：

1. **Prompt优化**：在 `medical_transplant_prompt.py` 中添加更多示例
2. **Schema扩展**：在 `medical_kg_schema.py` 中添加新的实体/关系类型
3. **后处理规则**：创建冲突解决规则

### Phase 6: 全量处理（2-3天）

#### 6.1 并行处理20000篇文章
创建20个并行任务（每个处理1000篇）：

```bash
# 生成20个配置文件
for i in {0..19}; do
cat > config_shard_${i}.py <<EOF
config = ProcessingConfig(
    ...
    total_shards_triple=20,
    current_shard_triple=${i},
    ...
)
EOF
done

# 并行运行（如果有多GPU）
for i in {0..19}; do
    CUDA_VISIBLE_DEVICES=$((i % 4)) python run_extraction.py --config config_shard_${i}.py &
done
```

#### 6.2 合并结果
```bash
# 合并所有CSV
cat output_shard_*/triple_nodes_*.csv > all_nodes.csv
cat output_shard_*/triple_edges_*.csv > all_edges.csv
```

## ⚠️ 关键注意事项

### 1. LLM成本控制
- 20000篇文章 × 平均250 tokens/abstract = 5M input tokens
- 假设每篇生成500 tokens输出 = 10M output tokens
- **估算成本**：根据您的LLM定价计算
- **建议**：使用本地部署的开源模型（Llama 3.1, Mistral）

### 2. 医学缩写消歧的特殊处理
**常见歧义**：
- PGD: Primary Graft Dysfunction vs Preimplantation Genetic Diagnosis
- LVAD: Left Ventricular Assist Device vs Low Voltage Activation Delay
- CAV: Cardiac Allograft Vasculopathy vs Central Arteriovenous

**解决方案**：
在 `medical_transplant_prompt.py` 的 system message 中添加：
```
"Context: All text is about heart transplantation. Disambiguate abbreviations accordingly."
```

### 3. 统计量提取的边界情况
**问题**：LLM可能提取错误的数值

**示例**：
- Text: "The study included 250 patients with OR=2.1"
- 错误提取：odds_ratio = 250

**解决方案**：
后处理验证：
```python
if 'Statistical_Metrics' in triple:
    if triple['Statistical_Metrics'].get('odds_ratio', 0) > 100:
        # 标记为需要人工审查
        triple['needs_review'] = True
```

### 4. Neo4j性能优化
**问题**：20000篇文章可能生成百万级三元组，查询变慢

**优化策略**：
```cypher
// 1. 为高频查询路径创建索引
CREATE INDEX rel_type_idx FOR ()-[r:RELATION]-() ON (r.type);

// 2. 使用图算法预计算
CALL gds.pageRank.write({
    nodeProjection: 'Entity',
    relationshipProjection: 'RELATION',
    writeProperty: 'pagerank'
})

// 3. 物化常用查询结果
CREATE VIEW high_risk_factors AS
MATCH (rf:Risk_Factor)-[r:RELATION {type: 'INCREASES_RISK_OF'}]->(c:Complication)
WHERE r.odds_ratio > 2.0
RETURN rf, r, c
```

### 5. 知识冲突解决策略
**场景**：不同文献对同一关系有不同的结论

**示例**：
- Paper A (2020, RCT): "Machine perfusion reduces PGD" (OR=0.5, p<0.01)
- Paper B (2018, Cohort): "Machine perfusion shows no benefit" (OR=0.9, p=0.3)

**解决策略**：
1. **证据强度排序**：RCT > Cohort，保留RCT的结论
2. **时间优先**：新研究 > 旧研究
3. **元分析优先**：如果有Meta-analysis，优先采用
4. **保留争议标记**：
```cypher
CREATE (rf:Risk_Factor {name: "Machine Perfusion"})
CREATE (c:Complication {name: "PGD"})
CREATE (rf)-[:MITIGATES {
    consensus: false,
    conflicting_evidence: ["pmid_12345", "pmid_67890"],
    latest_evidence: "pmid_12345"
}]->(c)
```

## 🔍 常见问题排查

### Q1: LLM输出格式不符合JSON Schema
**症状**：`json_repair.loads()` 抛出异常

**排查**：
```bash
# 检查原始LLM输出
cat test_output/kg_extraction/*.json | jq '.entity_relation_output' | head -5
```

**解决**：
- 调整 `max_new_tokens`（可能输出被截断）
- 增强 system prompt："You MUST output valid JSON, no explanation before or after"

### Q2: 大量实体被归一化为同一个UMLS CUI
**症状**：不同的实体（如"donor age"和"recipient age"）被映射到同一个CUI

**排查**：
```python
df = pd.read_csv("triple_nodes_normalized.csv")
duplicate_cuis = df.groupby('umls_cui').size().sort_values(ascending=False)
print(duplicate_cuis.head(20))
```

**解决**：
- 使用更细粒度的UMLS semantic types
- 添加上下文信息到QuickUMLS查询

### Q3: Neo4j导入速度很慢
**症状**：导入百万三元组需要数小时

**解决**：
```bash
# 使用 neo4j-admin import（比LOAD CSV快10倍）
neo4j-admin database import full \
    --nodes=Entity=triple_nodes.csv \
    --relationships=RELATION=triple_edges.csv \
    --delimiter=',' \
    --array-delimiter=';' \
    neo4j
```

## 📊 预期里程碑

| 阶段 | 时间 | 可交付成果 |
|------|------|-----------|
| Phase 1-2 | Day 1 | 100篇测试集的KG（验证pipeline） |
| Phase 3 | Day 2-3 | 归一化后的实体（UMLS映射） |
| Phase 4 | Day 4 | Neo4j中的测试KG（可查询） |
| Phase 5 | Day 5 | 质量报告 + 改进计划 |
| Phase 6 | Day 6-8 | 完整20000篇的KG |

## ✅ 成功标准

1. **覆盖率**：>80%的文章至少提取到5个三元组
2. **准确率**：人工抽查100个三元组，>90%正确
3. **归一化率**：>70%的实体成功映射到UMLS
4. **可用性**：Neo4j查询响应时间<2秒

## 🚀 下一步行动

1. [ ] 分享您的schema文件位置（我需要看具体定义）
2. [ ] 提供1-2篇文章JSON样例（验证格式兼容性）
3. [ ] 确认LLM选择（本地部署 vs API）
4. [ ] 确认Neo4j访问方式（本地 vs 云端）

完成这些后，我将为您生成：
- 定制化的配置文件
- 自动化脚本
- 质量验证工具
