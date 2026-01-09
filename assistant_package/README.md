# 助手运行包 - 知识抽取后半部分（12217-24432）

## 📦 包含文件

```
assistant_package/
├── README.md                    # 本文件
├── automated_kg_pipeline/
│   ├── assistant_extract.py    # 主抽取脚本
│   ├── config.yaml             # 配置文件（需要填写API key）
│   └── requirements.txt        # Python依赖
├── schemas/
│   └── chinese_medical_kg_schema.json  # 知识抽取Schema
├── data/
│   └── medical_abstracts/
│       └── heart_tx_all_merged_v8.json  # 数据文件（需要复制）
└── run.sh                       # 一键启动脚本

```

## ⚡ 快速启动（3步）

### 1. 安装依赖

```bash
cd automated_kg_pipeline
pip install -r requirements.txt
```

### 2. 配置API Key

编辑 `automated_kg_pipeline/config.yaml`，填写你的DeepSeek API key：

```yaml
llm:
  deepseek:
    api_key: "sk-your-api-key-here"  # ← 替换这里
    base_url: "https://yinli.one/v1"
    model: "deepseek-chat"
```

### 3. 启动抽取

```bash
# 后台运行
nohup python3 -u automated_kg_pipeline/assistant_extract.py > logs/assistant_extraction.log 2>&1 &

# 查看进度
tail -f logs/assistant_extraction.log

# 或使用监控脚本
bash 监控进度_助手.sh
```

---

## 📊 任务说明

- **你负责**: 文章 12217 ~ 24432（共12216篇，后半部分）
- **主端负责**: 文章 1 ~ 12216（共12216篇，前半部分）
- **总计**: 24432篇

### 为什么这样分？

1. **避免冲突**: 使用不同的检查点文件（`assistant_checkpoint.json`）
2. **并行加速**: 两边同时跑，速度翻倍
3. **独立缓存**: 共享 `cache/parsed_triples/` 和 `cache/llm_raw_outputs/`

---

## 🔧 详细说明

### 配置文件说明

`config.yaml` 关键配置项：

```yaml
llm:
  provider: "deepseek"
  deepseek:
    api_key: "sk-..."              # 你的API key
    base_url: "https://yinli.one/v1"
    model: "deepseek-chat"
    max_tokens: 2048
    temperature: 0.1

neo4j:
  uri: "bolt://localhost:7687"    # 暂时用不到，可以忽略
  username: "neo4j"
  password: "wunai1127"
  database: "htkg"

data:
  input_directory: "../data/medical_abstracts"
  field_mapping:
    text_field: "text"
    id_field: "id"
```

### 脚本功能

**assistant_extract.py** 自动处理：

- ✅ **断点续传**: 中断后重新运行自动继续
- ✅ **智能重试**: API限流/503错误自动等待重试
- ✅ **缓存复用**: 已处理的文章直接跳过
- ✅ **余额检测**: 余额不足时保存进度并停止

### 运行命令

```bash
# 方式1: 前台运行（测试用）
python3 automated_kg_pipeline/assistant_extract.py

# 方式2: 后台运行（推荐）
nohup python3 -u automated_kg_pipeline/assistant_extract.py > logs/assistant_extraction.log 2>&1 &

# 方式3: 使用screen（服务器推荐）
screen -S kg_extract
python3 automated_kg_pipeline/assistant_extract.py
# Ctrl+A+D 分离会话
# screen -r kg_extract 重新连接
```

### 监控进度

```bash
# 实时查看日志
tail -f logs/assistant_extraction.log

# 查看已处理数量
ls cache/parsed_triples/ | wc -l

# 查看检查点
cat cache/assistant_checkpoint.json | python3 -m json.tool

# 使用监控脚本（如果有）
bash 监控进度_助手.sh
```

---

## 📁 输出文件

### 1. 原始LLM输出

```
cache/llm_raw_outputs/
├── 41123465_raw.json
├── 41079538_raw.json
└── ...
```

每个文件包含：
- 文章ID
- 完整Prompt
- LLM原始回复
- 时间戳

### 2. 解析后的三元组

```
cache/parsed_triples/
├── 41123465_triples.json
├── 41079538_triples.json
└── ...
```

格式：
```json
{
  "entities": [
    {"name": "原发性移植物功能障碍", "type": "并发症", "properties": {}}
  ],
  "relations": [
    {"head": "高钾血症", "relation": "导致", "tail": "心脏骤停", "properties": {}}
  ]
}
```

### 3. 检查点文件

```
cache/assistant_checkpoint.json
```

格式：
```json
{
  "processed_ids": ["41123465", "41079538", ...],
  "last_index": 1523,
  "start_time": "2026-01-09T13:00:00"
}
```

---

## ⚠️ 常见问题

### 1. SSL证书错误

**错误**: `TLS_error:CERTIFICATE_VERIFY_FAILED`

**解决**: 脚本已禁用SSL验证（`verify=False`），如果还有问题：

```python
# 在assistant_extract.py中确认这行存在
http_client=httpx.Client(verify=False, timeout=60.0)
```

### 2. API限流（503错误）

**症状**: 日志显示 `HTTP/1.1 503 Service Unavailable`

**解决**: 脚本会自动重试，等待时间：1s → 3s → 5s → 10s → 30s → 60s → 2min → 5min → 10min

### 3. 余额不足

**症状**: 日志显示 `余额不足`

**解决**:
1. 充值DeepSeek账户
2. 重新运行相同命令，自动从断点继续

### 4. 进程意外中断

**解决**:
1. 检查日志：`tail -100 logs/assistant_extraction.log`
2. 查看检查点：`cat cache/assistant_checkpoint.json`
3. 重新运行脚本，自动从上次停止的地方继续

### 5. 内存不足

**症状**: `MemoryError` 或进程被killed

**解决**:
- 确保服务器有至少2GB可用内存
- 或修改脚本分批加载数据（如果需要我可以提供修改版本）

---

## 📈 预期时间和成本

### 处理速度

- 单篇文章: ~30秒（包含API调用+解析）
- 12216篇: 约 **102小时**（4.25天）
- 如果24小时不间断运行

### API成本

- 单篇文章: ~800 tokens（输入+输出）
- 12216篇: ~9,772,800 tokens ≈ 9.7M tokens
- DeepSeek价格: 0.001元/1K tokens
- **预计成本**: ~10元

### 建议

- 使用 `screen` 或 `nohup` 保持后台运行
- 定期检查进度（每小时）
- 确保服务器稳定、网络畅通

---

## 🔄 与主端合并

### 抽取完成后

两边抽取完成后，缓存文件会自动合并（因为使用同一个 `cache/` 目录）：

```bash
# 检查总数
ls cache/parsed_triples/ | wc -l
# 应该显示: 24432

# 检查是否有重复
ls cache/parsed_triples/ | sort | uniq -d
# 应该为空
```

### 导入Neo4j

等两边都完成后，由主端统一导入Neo4j：

```bash
# 主端运行
python3 import_to_neo4j.py
```

---

## 📞 联系与问题

如果遇到任何问题：

1. **检查日志**: `logs/assistant_extraction.log`
2. **查看检查点**: `cache/assistant_checkpoint.json`
3. **联系主端**: 分享日志文件和错误信息

---

## ✅ 检查清单

开始前确认：

- [ ] Python 3.8+ 已安装
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] API key 已配置（`config.yaml`）
- [ ] 数据文件已复制（`heart_tx_all_merged_v8.json`）
- [ ] 日志目录存在（`mkdir -p logs`）
- [ ] 缓存目录存在（`mkdir -p cache/{llm_raw_outputs,parsed_triples}`）

运行：

- [ ] 启动抽取脚本
- [ ] 确认日志正常输出
- [ ] 每小时检查一次进度

完成：

- [ ] 确认处理完12216篇
- [ ] 检查错误数（应该很少）
- [ ] 通知主端合并数据

---

## 🎯 预期结果

完成后你应该看到：

```
============================================================
当前进度
============================================================
处理文章: 12216/12216
完成度: 100.00%
总实体: ~105,000
总关系: ~88,000
错误数: <100
平均实体/文章: 8.6
平均关系/文章: 7.2

缓存位置:
  - 原始输出: cache/llm_raw_outputs/
  - 解析结果: cache/parsed_triples/
  - 检查点: cache/assistant_checkpoint.json
============================================================
```

文件数量：

```bash
ls cache/llm_raw_outputs/ | wc -l
# → 应该 ≥ 12216（包含主端的）

ls cache/parsed_triples/ | wc -l
# → 应该 ≥ 12216（包含主端的）
```

---

祝运行顺利！🚀
