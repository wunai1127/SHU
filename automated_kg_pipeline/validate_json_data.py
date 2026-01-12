#!/usr/bin/env python3
"""
验证 heart_tx_all_merged_v8.json 数据格式
"""

import json
import sys
from pathlib import Path
from collections import Counter

def validate_json_file(file_path: str):
    """验证JSON文件格式和内容"""

    print("="*60)
    print("JSON数据验证工具")
    print("="*60)
    print()

    # 检查文件是否存在
    if not Path(file_path).exists():
        print(f"❌ 文件不存在: {file_path}")
        print()
        print("请确认:")
        print("1. heart_tx_all_merged_v8.json.zip 已解压")
        print("2. JSON文件在正确的目录")
        return False

    print(f"✓ 文件存在: {file_path}")
    file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
    print(f"  文件大小: {file_size:.2f} MB")
    print()

    # 尝试加载JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ JSON格式正确")
    except json.JSONDecodeError as e:
        print(f"❌ JSON格式错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False

    # 检查数据结构
    if not isinstance(data, list):
        print(f"❌ 数据格式错误: 期望JSON数组，实际为 {type(data)}")
        return False

    print(f"✓ 数据类型正确: JSON数组")
    print(f"  文章数量: {len(data)}")
    print()

    if len(data) == 0:
        print("❌ 数据为空")
        return False

    # 分析字段结构
    print("─"*60)
    print("字段分析")
    print("─"*60)

    # 检查第一篇文章的字段
    first_article = data[0]
    print(f"第一篇文章字段: {list(first_article.keys())}")
    print()

    # 统计所有文章的字段
    all_fields = set()
    field_counts = Counter()

    for article in data[:min(100, len(data))]:  # 采样前100篇
        fields = set(article.keys())
        all_fields.update(fields)
        for field in fields:
            field_counts[field] += 1

    print(f"所有字段（前100篇采样）:")
    for field, count in field_counts.most_common():
        coverage = count / min(100, len(data)) * 100
        print(f"  - {field:20s} : {count:3d} / {min(100, len(data))} ({coverage:.1f}%)")
    print()

    # 检查必需字段
    print("─"*60)
    print("必需字段检查")
    print("─"*60)

    # 检查可能的文本字段
    text_field_candidates = ['abstract', 'text', 'content', 'summary', 'description']
    found_text_field = None

    for candidate in text_field_candidates:
        if candidate in first_article:
            found_text_field = candidate
            break

    if found_text_field:
        print(f"✓ 找到文本字段: '{found_text_field}'")
        sample_text = first_article[found_text_field]
        print(f"  示例内容: {sample_text[:100]}...")
        print(f"  文本长度: {len(sample_text)} 字符")
    else:
        print(f"⚠️  未找到常见文本字段 {text_field_candidates}")
        print(f"  第一篇文章内容:")
        for key, value in first_article.items():
            if isinstance(value, str) and len(value) > 50:
                print(f"    {key}: {value[:80]}...")
    print()

    # 检查ID字段
    id_field_candidates = ['pmid', 'id', 'article_id', 'doc_id', 'PMID']
    found_id_field = None

    for candidate in id_field_candidates:
        if candidate in first_article:
            found_id_field = candidate
            break

    if found_id_field:
        print(f"✓ 找到ID字段: '{found_id_field}'")
        print(f"  示例ID: {first_article[found_id_field]}")
    else:
        print(f"⚠️  未找到常见ID字段 {id_field_candidates}")
        print(f"  建议: 使用数组索引作为ID")
    print()

    # 生成配置建议
    print("="*60)
    print("配置建议")
    print("="*60)
    print()

    if found_text_field and found_id_field:
        print(f"请在 config.yaml 中配置:")
        print()
        print(f"data:")
        print(f"  field_mapping:")
        print(f"    text_field: \"{found_text_field}\"")
        print(f"    id_field: \"{found_id_field}\"")
        print()
        print("✓ 数据格式验证通过！可以开始构建知识图谱。")
    elif found_text_field:
        print(f"请在 config.yaml 中配置:")
        print()
        print(f"data:")
        print(f"  field_mapping:")
        print(f"    text_field: \"{found_text_field}\"")
        print(f"    id_field: \"id\"  # 将使用数组索引")
        print()
        print("⚠️  缺少ID字段，将使用数组索引作为ID")
    else:
        print("❌ 缺少必需的文本字段")
        print()
        print("请检查JSON结构，确保每个文章对象包含文本内容")
        return False

    print()
    print("="*60)
    print(f"验证完成 - {len(data)} 篇文章")
    print("="*60)

    return True

if __name__ == '__main__':
    # 默认路径
    default_path = "/home/user/SHU/data/medical_abstracts/heart_tx_all_merged_v8.json"

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path

    success = validate_json_file(file_path)
    sys.exit(0 if success else 1)
