# 知识图谱构建流程优化策略

## 一、并行处理架构（核心优化）

### 1.1 三层并行设计

```
┌─────────────────────────────────────────────────────────┐
│              20000篇文章 (Input Layer)                   │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌────────┐    ┌────────┐    ┌────────┐
   │Shard 1 │    │Shard 2 │ ...│Shard 20│  (Shard Layer)
   │1000篇  │    │1000篇  │    │1000篇  │
   └────┬───┘    └────┬───┘    └────┬───┘
        │             │             │
   ┌────▼────┐   ┌───▼─────┐  ┌───▼─────┐
   │GPU 0    │   │GPU 1    │  │GPU 2/3  │  (GPU Layer)
   │LLM抽取  │   │LLM抽取  │  │LLM抽取  │
   └────┬────┘   └────┬────┘  └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      ▼
            ┌──────────────────┐
            │  Neo4j批量导入    │       (Sink Layer)
            │  (5000/batch)    │
            └──────────────────┘
```

**实现代码**：

```python
# automated_kg_pipeline/parallel_processor.py
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

class ParallelKGBuilder:
    def __init__(self, num_gpus: int = 2):
        self.num_gpus = num_gpus
        self.num_workers = num_gpus * 2  # 每个GPU运行2个worker

    def process_shards_parallel(self, articles: List[Dict], shard_size: int = 1000):
        """并行处理分片"""
        shards = [articles[i:i+shard_size] for i in range(0, len(articles), shard_size)]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_shard, shard_id, shard, gpu_id % self.num_gpus): shard_id
                for shard_id, (shard, gpu_id) in enumerate(zip(shards, range(len(shards))))
            }

            results = []
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    shard_triples = future.result()
                    results.extend(shard_triples)
                    print(f"✓ Shard {shard_id} 完成: {len(shard_triples)} triples")
                except Exception as e:
                    print(f"✗ Shard {shard_id} 失败: {e}")

        return results

    def _process_shard(self, shard_id: int, articles: List[Dict], gpu_id: int):
        """处理单个分片（在独立进程中运行）"""
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        # 初始化LLM（每个进程独立加载）
        extractor = self._init_extractor(gpu_id)

        # 处理文章
        triples = []
        for article in articles:
            triples.extend(extractor.extract(article))

        # 保存分片结果到磁盘（避免内存爆炸）
        shard_file = f"/tmp/kg_shard_{shard_id}.json"
        with open(shard_file, 'w') as f:
            json.dump(triples, f)

        return shard_file  # 返回文件路径而非数据
```

### 1.2 估算性能

| 配置 | 单篇处理时间 | 总时间（20000篇） |
|------|------------|-----------------|
| **单GPU顺序** | 10秒 | 55.5小时 |
| **2 GPU并行** | 10秒 | 27.8小时 |
| **4 GPU并行** | 10秒 | 13.9小时 |
| **API (10 QPS)** | 1秒 | 33分钟 |

**推荐配置**：
- 如果有预算：使用DeepSeek API（成本约50-100元，30分钟完成）
- 如果本地GPU：4个GPU并行（约14小时完成）

---

## 二、缓存与增量更新

### 2.1 三级缓存设计

```python
# cache/cache_manager.py
import hashlib
import pickle
from pathlib import Path

class KGCache:
    """三级缓存管理器"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Level 1: LLM响应缓存（避免重复调用）
        self.llm_cache_dir = self.cache_dir / "llm_responses"
        self.llm_cache_dir.mkdir(exist_ok=True)

        # Level 2: 解析后的三元组缓存
        self.triple_cache_dir = self.cache_dir / "triples"
        self.triple_cache_dir.mkdir(exist_ok=True)

        # Level 3: Neo4j导入状态缓存
        self.import_state_file = self.cache_dir / "import_state.pkl"

    def get_llm_response(self, article_id: str, text_hash: str):
        """获取LLM缓存响应"""
        cache_key = f"{article_id}_{text_hash}"
        cache_file = self.llm_cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None

    def save_llm_response(self, article_id: str, text: str, response: str):
        """保存LLM响应"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        cache_key = f"{article_id}_{text_hash}"
        cache_file = self.llm_cache_dir / f"{cache_key}.json"

        with open(cache_file, 'w') as f:
            json.dump({"response": response}, f)

    def is_article_processed(self, article_id: str) -> bool:
        """检查文章是否已处理"""
        triple_file = self.triple_cache_dir / f"{article_id}.json"
        return triple_file.exists()

    def get_import_state(self) -> Dict:
        """获取导入状态"""
        if self.import_state_file.exists():
            with open(self.import_state_file, 'rb') as f:
                return pickle.load(f)
        return {"imported_articles": set(), "last_batch_id": 0}

    def update_import_state(self, article_ids: List[str], batch_id: int):
        """更新导入状态"""
        state = self.get_import_state()
        state["imported_articles"].update(article_ids)
        state["last_batch_id"] = batch_id

        with open(self.import_state_file, 'wb') as f:
            pickle.dump(state, f)
```

**增量更新流程**：

```python
def incremental_update(new_articles: List[Dict]):
    """增量更新KG（只处理新文章）"""
    cache = KGCache("/home/user/SHU/cache")

    # 过滤已处理的文章
    to_process = [
        article for article in new_articles
        if not cache.is_article_processed(article['id'])
    ]

    print(f"新增文章: {len(to_process)} / {len(new_articles)}")

    # 只处理新文章
    new_triples = extract_knowledge(to_process)

    # 增量导入Neo4j
    import_to_neo4j(new_triples)
```

---

## 三、Neo4j导入优化

### 3.1 批量导入 vs 实时导入

| 方法 | 速度 | 适用场景 |
|------|------|---------|
| **Cypher MERGE** | 1000节点/秒 | 实时更新 |
| **APOC批量** | 10000节点/秒 | 中等规模 |
| **neo4j-admin import** | 100000节点/秒 | 初次全量导入 |

**推荐方案**：

```bash
# 第一次全量导入：使用neo4j-admin（最快）
# 1. 停止Neo4j服务
sudo systemctl stop neo4j

# 2. 导出三元组为CSV
python export_to_csv.py --output /tmp/kg_csv/

# 3. 使用neo4j-admin导入
neo4j-admin database import full \
    --nodes=Entity=/tmp/kg_csv/entities.csv \
    --relationships=RELATION=/tmp/kg_csv/relations.csv \
    --overwrite-destination \
    heart_transplant_kg

# 4. 重启Neo4j
sudo systemctl start neo4j
```

**CSV格式规范**：

```csv
# entities.csv
id:ID,name,type:LABEL,properties
e1,"延长缺血时间","风险因子","{\"风险等级\":\"高\"}"
e2,"原发性移植物功能障碍","并发症","{\"发生率\":\"10-20%\"}"

# relations.csv
:START_ID,:END_ID,:TYPE,properties
e1,e2,导致,"{\"证据强度\":\"RCT\",\"优势比\":2.3,\"P值\":0.001}"
```

### 3.2 Neo4j配置优化

```conf
# /etc/neo4j/neo4j.conf

# 内存配置（假设服务器有32GB内存）
dbms.memory.heap.initial_size=8g
dbms.memory.heap.max_size=8g
dbms.memory.pagecache.size=16g

# 批量导入优化
dbms.transaction.timeout=300s
dbms.lock.acquisition.timeout=300s

# 并发配置
dbms.threads.worker_count=8

# 日志级别（生产环境降低日志量）
dbms.logs.query.enabled=false
```

---

## 四、质量保证机制

### 4.1 实时质量监控

```python
class QualityMonitor:
    """实时质量监控器"""

    def __init__(self):
        self.stats = {
            "low_confidence_triples": [],
            "invalid_relations": [],
            "outlier_articles": []
        }

    def check_triple_quality(self, triple: Dict, article_id: str):
        """检查单个三元组质量"""
        issues = []

        # 检查1: 置信度
        if triple.get('confidence', 1.0) < 0.7:
            issues.append("低置信度")
            self.stats["low_confidence_triples"].append(triple)

        # 检查2: Schema验证
        if not self._validate_schema(triple):
            issues.append("Schema不符")
            self.stats["invalid_relations"].append(triple)

        # 检查3: 统计量合理性
        if 'properties' in triple:
            props = triple['properties']
            if 'p_value' in props and props['p_value'] > 1.0:
                issues.append("P值异常")

        return issues

    def check_article_quality(self, article_id: str, triples: List[Dict]):
        """检查单篇文章质量"""
        num_entities = len([t for t in triples if t['type'] == 'entity'])
        num_relations = len([t for t in triples if t['type'] == 'relation'])

        # 异常检测
        if num_entities < 5:
            self.stats["outlier_articles"].append({
                "article_id": article_id,
                "issue": "实体数过少",
                "count": num_entities
            })

        if num_relations == 0:
            self.stats["outlier_articles"].append({
                "article_id": article_id,
                "issue": "无关系抽取"
            })

    def generate_quality_report(self) -> Dict:
        """生成质量报告"""
        return {
            "低置信度三元组数": len(self.stats["low_confidence_triples"]),
            "Schema不符三元组数": len(self.stats["invalid_relations"]),
            "异常文章数": len(self.stats["outlier_articles"]),
            "异常文章详情": self.stats["outlier_articles"][:10]  # 前10个
        }
```

### 4.2 自动修复机制

```python
class AutoFixer:
    """自动修复常见错误"""

    @staticmethod
    def fix_abbreviation(entity_name: str, context: str) -> str:
        """缩写展开"""
        abbr_map = {
            "PGD": "Primary Graft Dysfunction",
            "PVR": "Pulmonary Vascular Resistance",
            "ISHLT": "International Society for Heart and Lung Transplantation",
            "LVEF": "Left Ventricular Ejection Fraction"
        }
        return abbr_map.get(entity_name, entity_name)

    @staticmethod
    def fix_duplicate_entities(triples: List[Dict]) -> List[Dict]:
        """合并重复实体"""
        entity_map = {}

        for triple in triples:
            if triple['type'] == 'entity':
                name = triple['name']
                # 标准化名称
                normalized_name = AutoFixer._normalize_entity_name(name)

                if normalized_name not in entity_map:
                    entity_map[normalized_name] = triple
                else:
                    # 合并属性
                    entity_map[normalized_name]['properties'].update(triple.get('properties', {}))

        # 更新关系中的实体引用
        fixed_triples = list(entity_map.values())

        for triple in triples:
            if triple['type'] == 'relation':
                triple['head'] = AutoFixer._normalize_entity_name(triple['head'])
                triple['tail'] = AutoFixer._normalize_entity_name(triple['tail'])
                fixed_triples.append(triple)

        return fixed_triples

    @staticmethod
    def _normalize_entity_name(name: str) -> str:
        """实体名称标准化"""
        # 转小写
        normalized = name.lower().strip()
        # 移除标点
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # 展开缩写
        normalized = AutoFixer.fix_abbreviation(normalized, "")
        return normalized
```

---

## 五、性能瓶颈识别与解决

### 5.1 Profile分析

```python
import cProfile
import pstats

def profile_pipeline():
    """性能分析"""
    profiler = cProfile.Profile()
    profiler.enable()

    # 运行流水线
    builder = AutoKGBuilder('config.yaml')
    builder.run()

    profiler.disable()

    # 输出报告
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 前20个最慢函数
```

**常见瓶颈与解决方案**：

| 瓶颈 | 现象 | 解决方案 |
|------|------|---------|
| **LLM推理慢** | GPU利用率100% | 增加GPU数量 / 使用API / 量化模型 |
| **JSON解析慢** | CPU利用率高 | 使用ujson / orjson替代json |
| **Neo4j写入慢** | 网络延迟高 | 批量导入 / 使用本地Neo4j |
| **内存不足** | OOM错误 | 减小batch_size / 使用流式处理 |

---

## 六、优化效果对比

### 基线配置 vs 优化配置

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| **总处理时间** | 55.5小时 | 14小时 | 4x |
| **GPU利用率** | 25% | 90% | 3.6x |
| **内存占用** | 32GB | 16GB | 0.5x |
| **Neo4j导入速度** | 1000节点/秒 | 100000节点/秒 | 100x |
| **可恢复性** | 无 | 断点续传 | ∞ |

---

## 七、一键优化脚本

```python
# automated_kg_pipeline/optimize_and_run.py

def optimize_config(config_path: str) -> Dict:
    """自动优化配置"""
    config = load_config(config_path)

    # 检测GPU数量
    num_gpus = torch.cuda.device_count()
    config['extraction']['num_workers'] = num_gpus * 2

    # 检测内存大小
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)

    if total_memory_gb > 64:
        config['extraction']['batch_size'] = 16
    elif total_memory_gb > 32:
        config['extraction']['batch_size'] = 8
    else:
        config['extraction']['batch_size'] = 4

    # 检测Neo4j连接速度
    latency = test_neo4j_latency(config['neo4j'])
    if latency > 100:  # >100ms说明是远程连接
        config['neo4j']['batch_import']['batch_size'] = 10000
    else:
        config['neo4j']['batch_import']['batch_size'] = 5000

    return config

if __name__ == '__main__':
    # 自动优化配置
    config = optimize_config('config_template.yaml')

    # 运行流水线
    builder = AutoKGBuilder(config)
    builder.run()
```

---

## 八、执行时间表（20000篇文章）

| 阶段 | 时间 | 可优化项 |
|------|------|---------|
| **数据加载** | 5分钟 | 使用jsonl格式 |
| **LLM抽取** | 13小时 | **核心瓶颈** - 多GPU并行 |
| **质量过滤** | 10分钟 | 编译Cython加速 |
| **Neo4j导入** | 30分钟 | 使用neo4j-admin |
| **验证报告** | 5分钟 | - |
| **总计** | ~14小时 | |

**优化优先级**：
1. ⚡ **最高优先级**：LLM抽取并行化（节省40小时）
2. 🔥 **高优先级**：Neo4j批量导入（节省2小时）
3. 🌟 **中优先级**：启用缓存机制（重跑节省100%时间）
