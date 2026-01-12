#!/usr/bin/env python3
"""
简化的知识抽取测试脚本 - 用于快速验证API和抽取流程
"""

import json
import sys
import time
from pathlib import Path
from openai import OpenAI
import yaml

# 加载配置
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 初始化OpenAI客户端（v1.0+）
import httpx
client = OpenAI(
    api_key=config['llm']['deepseek']['api_key'],
    base_url=config['llm']['deepseek']['base_url'],
    http_client=httpx.Client(verify=False)  # 禁用SSL验证（代理证书问题）
)

# 加载Schema
schema_path = Path(__file__).parent.parent / "schemas/chinese_medical_kg_schema.json"
with open(schema_path, 'r') as f:
    schema = json.load(f)

entity_types = list(schema['entity_types'].keys())
relation_types = list(schema['relation_types'].keys())

def extract_knowledge(text: str, article_id: str, max_retries=3):
    """从单篇文章中抽取知识"""

    # 构造Prompt
    prompt = f"""请从以下医学文献中抽取心脏移植相关的知识三元组。

**实体类型**: {', '.join(entity_types)}
**关系类型**: {', '.join(relation_types)}

**文本**:
{text[:2000]}

**要求**:
1. 抽取所有相关的实体和关系
2. 如果有统计数据（优势比、p值），必须包含
3. 如果有证据强度信息（RCT、队列研究等），必须标注
4. 输出严格的JSON格式

**输出格式**:
{{
  "entities": [
    {{"name": "...", "type": "...", "properties": {{}}}}
  ],
  "relations": [
    {{"head": "...", "relation": "...", "tail": "...", "properties": {{}}}}
  ]
}}

请输出JSON:
"""

    for attempt in range(max_retries):
        try:
            # 限流：每次请求间隔1秒
            time.sleep(1)

            response = client.chat.completions.create(
                model=config['llm']['deepseek']['model'],
                messages=[
                    {"role": "system", "content": "你是医学信息抽取专家，专注于心脏移植领域。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config['llm']['deepseek']['max_tokens'],
                temperature=config['llm']['deepseek']['temperature']
            )

            result = response.choices[0].message.content

            # 保存原始输出
            cache_dir = Path(__file__).parent.parent / "cache/llm_raw_outputs"
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / f"{article_id}_raw.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "article_id": article_id,
                    "prompt": prompt,
                    "response": result,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": config['llm']['deepseek']['model']
                }, f, ensure_ascii=False, indent=2)

            # 解析JSON
            # 提取JSON部分
            start = result.find('{')
            end = result.rfind('}') + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                parsed = json.loads(json_str)

                # 保存解析后的结果
                parsed_dir = Path(__file__).parent.parent / "cache/parsed_triples"
                parsed_dir.mkdir(parents=True, exist_ok=True)

                parsed_file = parsed_dir / f"{article_id}_triples.json"
                with open(parsed_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

                return parsed
            else:
                print(f"⚠️  无法解析JSON，原始输出：{result[:200]}")
                return {"entities": [], "relations": []}

        except Exception as e:
            print(f"❌ 第{attempt+1}次尝试失败: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"   等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"❌ 达到最大重试次数，跳过文章 {article_id}")
                return {"entities": [], "relations": []}

def main():
    # 加载数据
    data_path = Path(__file__).parent.parent / "data/medical_abstracts/heart_tx_all_merged_v8.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 测试模式：只处理前10篇
    test_mode = "--test" in sys.argv or "--test-mode" in sys.argv
    max_articles = 10 if test_mode else len(data)

    print("="*60)
    print("知识抽取测试")
    print("="*60)
    print(f"总文章数: {len(data)}")
    print(f"处理数量: {max_articles}")
    print(f"模式: {'测试模式' if test_mode else '全量模式'}")
    print("="*60)
    print()

    stats = {
        "processed": 0,
        "entities": 0,
        "relations": 0,
        "errors": 0
    }

    for i, article in enumerate(data[:max_articles]):
        article_id = str(article['id'])
        text = article['text']

        print(f"\n[{i+1}/{max_articles}] 处理文章: {article_id}")
        print(f"  文本长度: {len(text)} 字符")

        result = extract_knowledge(text, article_id)

        if result:
            entity_count = len(result.get('entities', []))
            relation_count = len(result.get('relations', []))

            print(f"  ✓ 抽取成功: {entity_count} 实体, {relation_count} 关系")

            stats['processed'] += 1
            stats['entities'] += entity_count
            stats['relations'] += relation_count
        else:
            print(f"  ✗ 抽取失败")
            stats['errors'] += 1

        # 每5篇显示一次统计
        if (i + 1) % 5 == 0:
            print(f"\n--- 进度统计 ---")
            print(f"已处理: {stats['processed']}/{max_articles}")
            print(f"总实体: {stats['entities']}")
            print(f"总关系: {stats['relations']}")
            print(f"错误数: {stats['errors']}")
            print()

    # 最终统计
    print("\n" + "="*60)
    print("抽取完成")
    print("="*60)
    print(f"处理文章: {stats['processed']}/{max_articles}")
    print(f"总实体: {stats['entities']}")
    print(f"总关系: {stats['relations']}")
    print(f"错误数: {stats['errors']}")
    print(f"平均实体/文章: {stats['entities']/max(stats['processed'], 1):.1f}")
    print(f"平均关系/文章: {stats['relations']/max(stats['processed'], 1):.1f}")
    print()
    print(f"缓存位置:")
    print(f"  - 原始输出: cache/llm_raw_outputs/")
    print(f"  - 解析结果: cache/parsed_triples/")
    print("="*60)

if __name__ == '__main__':
    main()
