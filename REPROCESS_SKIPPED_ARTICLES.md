# 重新处理跳过文章的方法

## 方法1：清理检查点，重新从头抽（推荐）

如果跳过的文章不多（<100篇），最简单的方法是删除这些文章的检查点记录：

```python
import json

# 读取检查点
with open('cache/extraction_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)

# 要重新处理的文章ID列表
skipped_ids = [
    'pubmed_1650', 'pubmed_1654', 'pubmed_1655',
    'pubmed_1942', 'pubmed_1943', 'pubmed_1944', 'pubmed_1945',
    'pubmed_1946', 'pubmed_1947', 'pubmed_1948', 'pubmed_1949',
    'pubmed_1950', 'pubmed_1951', 'pubmed_1952', 'pubmed_1953',
    'pubmed_1954', 'pubmed_1955', 'pubmed_1956', 'pubmed_1957',
    'pubmed_1958', 'pubmed_1959'
]

# 从processed_ids中删除
for article_id in skipped_ids:
    if article_id in checkpoint['processed_ids']:
        checkpoint['processed_ids'].remove(article_id)

# 保存
with open('cache/extraction_checkpoint.json', 'w') as f:
    json.dump(checkpoint, f, indent=2)

print(f"✓ 已从检查点删除 {len(skipped_ids)} 篇文章")
print("重新运行抽取脚本，这些文章会被重新处理")
```

## 方法2：手动删除缓存文件

如果某些文章虽然"processed"但实际是空的（entities=[], relations=[]）：

```bash
# 查找空结果文件
find cache/parsed_triples/ -name "*.json" -exec sh -c 'grep -q "\"entities\": \[\]" "$1" && grep -q "\"relations\": \[\]" "$1" && echo "$1"' _ {} \;

# 删除空结果文件
find cache/parsed_triples/ -name "*.json" -exec sh -c 'grep -q "\"entities\": \[\]" "$1" && grep -q "\"relations\": \[\]" "$1" && rm "$1"' _ {} \;
```

## 方法3：从日志提取跳过列表（自动化）

从日志中提取所有跳过的文章ID：

```bash
grep "跳过文章" logs/main_extraction.log | grep -oE "pubmed_[0-9]+" | sort -u > skipped_articles.txt

# 生成Python清理脚本
cat > clean_skipped.py << 'EOF'
import json

with open('skipped_articles.txt', 'r') as f:
    skipped_ids = [line.strip() for line in f]

with open('cache/extraction_checkpoint.json', 'r') as f:
    checkpoint = json.load(f)

original_count = len(checkpoint['processed_ids'])
checkpoint['processed_ids'] = [id for id in checkpoint['processed_ids'] if id not in skipped_ids]
removed = original_count - len(checkpoint['processed_ids'])

with open('cache/extraction_checkpoint.json', 'w') as f:
    json.dump(checkpoint, f, indent=2)

print(f"✓ 从检查点删除了 {removed} 篇跳过的文章")
EOF

python3 clean_skipped.py
```

## 注意事项

1. **备份检查点**：修改前先备份
   ```bash
   cp cache/extraction_checkpoint.json cache/extraction_checkpoint.json.backup
   ```

2. **确认改进代码生效**：先确认新代码已运行，timeout处理改进

3. **重新运行**：清理后重新运行脚本，会自动处理之前跳过的文章

4. **验证结果**：处理完后检查这些文章是否有实体和关系
   ```bash
   cat cache/parsed_triples/pubmed_1650_triples.json
   ```
