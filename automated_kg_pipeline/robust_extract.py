#!/usr/bin/env python3
"""
健壮的知识抽取脚本 - 支持断点续传和API限流处理
Features:
- 智能重试策略（指数退避：1s → 10min）
- API限流检测和自适应等待
- 余额不足检测和优雅关闭
- 基于检查点的恢复能力
- 自动使用缓存结果
"""

import json
import sys
import time
from pathlib import Path
from openai import OpenAI
import yaml
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/user/SHU/logs/robust_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 加载配置
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 初始化OpenAI客户端
import httpx
client = OpenAI(
    api_key=config['llm']['deepseek']['api_key'],
    base_url=config['llm']['deepseek']['base_url'],
    http_client=httpx.Client(verify=False, timeout=60.0)
)

# 加载Schema
schema_path = Path(__file__).parent.parent / "schemas/chinese_medical_kg_schema.json"
with open(schema_path, 'r') as f:
    schema = json.load(f)

entity_types = list(schema['entity_types'].keys())
relation_types = list(schema['relation_types'].keys())

# 进度检查点文件
CHECKPOINT_FILE = Path(__file__).parent.parent / "cache/extraction_checkpoint.json"


def load_checkpoint():
    """加载检查点"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_ids": [], "last_index": 0, "start_time": None}


def save_checkpoint(checkpoint):
    """保存检查点"""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def is_rate_limit_error(error_msg: str) -> bool:
    """判断是否为API限流错误"""
    rate_limit_keywords = [
        'rate_limit', 'rate limit', 'too many requests',
        'quota', '429', 'RateLimitError', '503'
    ]
    error_str = str(error_msg).lower()
    return any(keyword in error_str for keyword in rate_limit_keywords)


def is_quota_exceeded_error(error_msg: str) -> bool:
    """判断是否为余额不足"""
    quota_keywords = [
        'insufficient', 'balance', 'quota exceeded',
        '余额不足', 'insufficient_quota'
    ]
    error_str = str(error_msg).lower()
    return any(keyword in error_str for keyword in quota_keywords)


def extract_knowledge(text: str, article_id: str):
    """从单篇文章中抽取知识 - 带智能重试"""

    # 检查缓存
    parsed_file = Path(__file__).parent.parent / f"cache/parsed_triples/{article_id}_triples.json"
    if parsed_file.exists():
        logger.info(f"  ✓ 使用缓存结果")
        with open(parsed_file, 'r') as f:
            return json.load(f)

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

    # 智能重试策略：逐步增加等待时间
    retry_delays = [1, 3, 5, 10, 30, 60, 120, 300, 600]  # 最长等10分钟

    for attempt in range(len(retry_delays)):
        try:
            # 基础限流：每次请求前等待1秒
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

            # 保存原始输出（关键：避免重复API调用）
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
            start = result.find('{')
            end = result.rfind('}') + 1
            if start != -1 and end > start:
                json_str = result[start:end]
                parsed = json.loads(json_str)

                # 保存解析后的结果
                parsed_dir = Path(__file__).parent.parent / "cache/parsed_triples"
                parsed_dir.mkdir(parents=True, exist_ok=True)

                with open(parsed_file, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)

                return parsed
            else:
                logger.warning(f"⚠️  无法解析JSON，原始输出：{result[:200]}")
                return {"entities": [], "relations": []}

        except Exception as e:
            error_msg = str(e)

            # 余额不足 - 立即停止并保存进度
            if is_quota_exceeded_error(error_msg):
                logger.error(f"❌ 余额不足！已保存进度，充值后可继续运行")
                logger.error(f"   错误信息: {error_msg}")
                raise Exception("QUOTA_EXCEEDED")

            # API限流或503错误 - 长时间等待后重试
            if is_rate_limit_error(error_msg):
                if attempt < len(retry_delays) - 1:
                    wait_time = retry_delays[attempt]
                    logger.warning(f"⚠️  API限流/503错误，等待{wait_time}秒后重试 (尝试 {attempt+1}/{len(retry_delays)})")
                    logger.warning(f"   错误: {error_msg[:200]}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ 达到最大重试次数，跳过文章 {article_id}")
                    return {"entities": [], "relations": []}

            # 其他错误 - 快速重试
            else:
                if attempt < 3:
                    wait_time = 2 ** attempt
                    logger.warning(f"⚠️  请求失败，{wait_time}秒后重试: {error_msg[:100]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ 跳过文章 {article_id}: {error_msg[:100]}")
                    return {"entities": [], "relations": []}

    return {"entities": [], "relations": []}


def main():
    # 加载数据
    data_path = Path(__file__).parent.parent / "data/medical_abstracts/heart_tx_all_merged_v8.json"
    with open(data_path, 'r') as f:
        data = json.load(f)

    # 加载检查点
    checkpoint = load_checkpoint()
    processed_ids = set(checkpoint['processed_ids'])

    if checkpoint['start_time'] is None:
        checkpoint['start_time'] = datetime.now().isoformat()

    total = len(data)
    logger.info("="*60)
    logger.info("健壮知识抽取 - 支持断点续传和限流处理")
    logger.info("="*60)
    logger.info(f"总文章数: {total}")
    logger.info(f"已处理: {len(processed_ids)}")
    logger.info(f"剩余: {total - len(processed_ids)}")
    logger.info(f"开始时间: {checkpoint['start_time']}")
    logger.info("="*60)

    stats = {
        "processed": len(processed_ids),
        "entities": 0,
        "relations": 0,
        "errors": 0
    }

    try:
        for i, article in enumerate(data):
            article_id = str(article['id'])

            # 跳过已处理
            if article_id in processed_ids:
                continue

            text = article['text']

            logger.info(f"\n[{i+1}/{total}] 处理文章: {article_id}")
            logger.info(f"  文本长度: {len(text)} 字符")

            result = extract_knowledge(text, article_id)

            if result:
                entity_count = len(result.get('entities', []))
                relation_count = len(result.get('relations', []))

                logger.info(f"  ✓ 抽取成功: {entity_count} 实体, {relation_count} 关系")

                stats['processed'] += 1
                stats['entities'] += entity_count
                stats['relations'] += relation_count

                # 更新检查点
                processed_ids.add(article_id)
                checkpoint['processed_ids'] = list(processed_ids)
                checkpoint['last_index'] = i
                save_checkpoint(checkpoint)
            else:
                logger.warning(f"  ✗ 抽取失败")
                stats['errors'] += 1

            # 每10篇显示统计
            if stats['processed'] % 10 == 0:
                logger.info(f"\n--- 进度统计 ---")
                logger.info(f"已处理: {stats['processed']}/{total}")
                logger.info(f"总实体: {stats['entities']}")
                logger.info(f"总关系: {stats['relations']}")
                logger.info(f"错误数: {stats['errors']}")
                logger.info(f"完成度: {stats['processed']/total*100:.2f}%")

    except KeyboardInterrupt:
        logger.info("\n⚠️  用户中断，已保存进度")
    except Exception as e:
        if "QUOTA_EXCEEDED" in str(e):
            logger.error("\n⚠️  余额不足，已保存进度。充值后运行相同命令即可继续")
        else:
            logger.error(f"\n❌ 意外错误: {e}")
    finally:
        # 最终统计
        logger.info("\n" + "="*60)
        logger.info("当前进度")
        logger.info("="*60)
        logger.info(f"处理文章: {stats['processed']}/{total}")
        logger.info(f"完成度: {stats['processed']/total*100:.2f}%")
        logger.info(f"总实体: {stats['entities']}")
        logger.info(f"总关系: {stats['relations']}")
        logger.info(f"错误数: {stats['errors']}")
        if stats['processed'] > 0:
            logger.info(f"平均实体/文章: {stats['entities']/stats['processed']:.1f}")
            logger.info(f"平均关系/文章: {stats['relations']/stats['processed']:.1f}")
        logger.info("\n缓存位置:")
        logger.info(f"  - 原始输出: cache/llm_raw_outputs/")
        logger.info(f"  - 解析结果: cache/parsed_triples/")
        logger.info(f"  - 检查点: {CHECKPOINT_FILE}")
        logger.info("="*60)


if __name__ == '__main__':
    main()
