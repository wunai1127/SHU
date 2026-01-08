#!/usr/bin/env python3
"""
心脏移植知识图谱全自动构建流水线
运行方式: python auto_kg_builder.py --config config.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import yaml
from tqdm import tqdm

# 添加AutoSchemaKG路径
sys.path.insert(0, str(Path(__file__).parent.parent / "AutoSchemaKG"))

from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from neo4j import GraphDatabase
import openai


class AutoKGBuilder:
    """全自动知识图谱构建器"""

    def __init__(self, config_path: str):
        """初始化构建器"""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_llm()
        self._setup_neo4j()
        self.stats = {
            "start_time": datetime.now(),
            "articles_processed": 0,
            "entities_extracted": 0,
            "relations_extracted": 0,
            "errors": []
        }

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _setup_logging(self):
        """配置日志"""
        log_file = self.config['logging']['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== 知识图谱构建流水线启动 ===")

    def _setup_llm(self):
        """配置LLM"""
        provider = self.config['llm']['provider']
        self.logger.info(f"配置LLM: {provider}")

        if provider == "openai":
            openai.api_key = self.config['llm']['openai']['api_key']
            openai.api_base = self.config['llm']['openai']['base_url']
            self.llm_config = self.config['llm']['openai']

        elif provider == "deepseek":
            openai.api_key = self.config['llm']['deepseek']['api_key']
            openai.api_base = self.config['llm']['deepseek']['base_url']
            self.llm_config = self.config['llm']['deepseek']

        elif provider == "local":
            self.llm_config = self.config['llm']['local']
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")

    def _setup_neo4j(self):
        """配置Neo4j连接"""
        neo4j_config = self.config['neo4j']
        self.logger.info(f"连接Neo4j: {neo4j_config['uri']}")

        self.neo4j_driver = GraphDatabase.driver(
            neo4j_config['uri'],
            auth=(neo4j_config['username'], neo4j_config['password'])
        )

        # 测试连接
        with self.neo4j_driver.session(database=neo4j_config['database']) as session:
            result = session.run("RETURN 1 AS test")
            assert result.single()['test'] == 1
            self.logger.info("Neo4j连接成功")

        # 创建索引和约束
        self._create_neo4j_schema()

    def _create_neo4j_schema(self):
        """创建Neo4j索引和约束"""
        self.logger.info("创建Neo4j索引和约束...")
        database = self.config['neo4j']['database']

        with self.neo4j_driver.session(database=database) as session:
            # 创建约束
            for constraint in self.config['neo4j']['constraints']:
                try:
                    query = f"""
                    CREATE CONSTRAINT IF NOT EXISTS
                    FOR (n:{constraint['label']})
                    REQUIRE n.{constraint['property']} IS UNIQUE
                    """
                    session.run(query)
                    self.logger.info(f"创建约束: {constraint['label']}.{constraint['property']}")
                except Exception as e:
                    self.logger.warning(f"约束创建失败: {e}")

            # 创建索引
            for index in self.config['neo4j']['indexes']:
                for prop in index['properties']:
                    try:
                        query = f"""
                        CREATE INDEX IF NOT EXISTS
                        FOR (n:{index['label']})
                        ON (n.{prop})
                        """
                        session.run(query)
                        self.logger.info(f"创建索引: {index['label']}.{prop}")
                    except Exception as e:
                        self.logger.warning(f"索引创建失败: {e}")

    def load_data(self) -> List[Dict]:
        """加载输入数据"""
        self.logger.info("加载输入数据...")
        data_config = self.config['data']
        input_dir = Path(data_config['input_directory'])
        pattern = data_config['filename_pattern']

        articles = []
        for file_path in input_dir.glob(pattern):
            self.logger.info(f"读取文件: {file_path}")

            with open(file_path, 'r', encoding=data_config['encoding']) as f:
                if data_config['expected_format'] == 'jsonl':
                    for line in f:
                        if line.strip():
                            articles.append(json.loads(line))
                else:
                    articles.extend(json.load(f))

        self.logger.info(f"共加载 {len(articles)} 篇文章")
        return articles

    def extract_knowledge(self, articles: List[Dict]) -> List[Dict]:
        """使用LLM抽取知识三元组"""
        self.logger.info("开始知识抽取...")
        extraction_config = self.config['extraction']

        # 加载schema
        with open(self.config['schema']['schema_file'], 'r', encoding='utf-8') as f:
            schema = json.load(f)

        all_triples = []

        # 处理每篇文章
        for article in tqdm(articles, desc="抽取三元组", disable=not self.config['logging']['enable_progress_bar']):
            try:
                # 构造LLM prompt
                text = article[self.config['data']['field_mapping']['text_field']]
                article_id = article[self.config['data']['field_mapping']['id_field']]

                triples = self._extract_triples_from_text(text, article_id, schema)

                # 质量过滤
                filtered_triples = self._filter_triples(triples, extraction_config)

                all_triples.extend(filtered_triples)
                self.stats['articles_processed'] += 1
                self.stats['entities_extracted'] += len([t for t in filtered_triples if t['type'] == 'entity'])
                self.stats['relations_extracted'] += len([t for t in filtered_triples if t['type'] == 'relation'])

            except Exception as e:
                self.logger.error(f"文章 {article_id} 抽取失败: {e}")
                self.stats['errors'].append({
                    'article_id': article_id,
                    'error': str(e)
                })

        self.logger.info(f"抽取完成: {len(all_triples)} 个三元组")
        return all_triples

    def _extract_triples_from_text(self, text: str, article_id: str, schema: Dict) -> List[Dict]:
        """从单篇文本抽取三元组（调用LLM）"""

        # 构造prompt
        prompt = self._build_extraction_prompt(text, schema)

        # 调用LLM
        if self.config['llm']['provider'] in ['openai', 'deepseek']:
            response = openai.ChatCompletion.create(
                model=self.llm_config['model'],
                messages=[
                    {"role": "system", "content": "你是医学信息抽取专家，专注于心脏移植领域。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.llm_config['max_tokens'],
                temperature=self.llm_config['temperature']
            )
            result_text = response.choices[0].message.content
        else:
            # 本地模型调用（需要集成vLLM或transformers）
            result_text = self._call_local_llm(prompt)

        # 解析LLM输出
        triples = self._parse_llm_output(result_text, article_id)
        return triples

    def _build_extraction_prompt(self, text: str, schema: Dict) -> str:
        """构造抽取prompt"""
        entity_types = ", ".join(schema['entity_types'].keys())
        relation_types = ", ".join(schema['relation_types'].keys())

        prompt = f"""请从以下医学文献摘要中抽取知识三元组。

**实体类型**: {entity_types}
**关系类型**: {relation_types}

**文本**:
{text}

**要求**:
1. 抽取所有相关的实体和关系
2. 如果文中有统计数据（优势比、p值），必须包含
3. 如果有证据强度信息（RCT、队列研究等），必须标注
4. 输出JSON格式：
[
  {{"type": "entity", "name": "...", "entity_type": "...", "properties": {{...}}}},
  {{"type": "relation", "head": "...", "relation_type": "...", "tail": "...", "properties": {{...}}}}
]

请输出JSON:
"""
        return prompt

    def _parse_llm_output(self, output: str, article_id: str) -> List[Dict]:
        """解析LLM输出的JSON"""
        try:
            # 提取JSON部分
            start = output.find('[')
            end = output.rfind(']') + 1
            json_str = output[start:end]

            triples = json.loads(json_str)

            # 添加来源信息
            for triple in triples:
                triple['source_article'] = article_id

            return triples
        except Exception as e:
            self.logger.error(f"解析LLM输出失败: {e}")
            return []

    def _call_local_llm(self, prompt: str) -> str:
        """调用本地LLM（需要实现）"""
        # TODO: 集成vLLM或transformers
        raise NotImplementedError("本地LLM调用尚未实现，请使用API模式")

    def _filter_triples(self, triples: List[Dict], config: Dict) -> List[Dict]:
        """质量过滤"""
        filtered = []
        for triple in triples:
            # 置信度过滤
            confidence = triple.get('confidence', 1.0)
            if confidence < config['min_confidence']:
                continue

            # Schema验证
            if config.get('enable_relation_validation', False):
                if not self._validate_triple_schema(triple):
                    continue

            filtered.append(triple)

        # 限制每篇文章的三元组数量
        max_triples = config['max_triples_per_article']
        if len(filtered) > max_triples:
            filtered = sorted(filtered, key=lambda x: x.get('confidence', 1.0), reverse=True)[:max_triples]

        return filtered

    def _validate_triple_schema(self, triple: Dict) -> bool:
        """验证三元组是否符合schema约束"""
        # TODO: 实现schema验证逻辑
        return True

    def import_to_neo4j(self, triples: List[Dict]):
        """导入三元组到Neo4j"""
        self.logger.info("导入到Neo4j...")
        database = self.config['neo4j']['database']
        batch_size = self.config['neo4j']['batch_import']['batch_size']

        # 分离实体和关系
        entities = [t for t in triples if t['type'] == 'entity']
        relations = [t for t in triples if t['type'] == 'relation']

        with self.neo4j_driver.session(database=database) as session:
            # 批量创建实体节点
            for i in tqdm(range(0, len(entities), batch_size), desc="导入实体"):
                batch = entities[i:i+batch_size]
                session.execute_write(self._create_entity_batch, batch)

            # 批量创建关系
            for i in tqdm(range(0, len(relations), batch_size), desc="导入关系"):
                batch = relations[i:i+batch_size]
                session.execute_write(self._create_relation_batch, batch)

        self.logger.info(f"导入完成: {len(entities)} 实体, {len(relations)} 关系")

    @staticmethod
    def _create_entity_batch(tx, entities):
        """批量创建实体节点"""
        query = """
        UNWIND $entities AS entity
        MERGE (n:Entity {name: entity.name, type: entity.entity_type})
        SET n += entity.properties
        SET n.source_article = entity.source_article
        """
        tx.run(query, entities=entities)

    @staticmethod
    def _create_relation_batch(tx, relations):
        """批量创建关系"""
        query = """
        UNWIND $relations AS rel
        MATCH (head:Entity {name: rel.head})
        MATCH (tail:Entity {name: rel.tail})
        CALL apoc.create.relationship(head, rel.relation_type, rel.properties, tail) YIELD rel AS r
        RETURN count(r)
        """
        tx.run(query, relations=relations)

    def generate_report(self):
        """生成构建报告"""
        self.logger.info("生成构建报告...")

        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = str(self.stats['end_time'] - self.stats['start_time'])

        report_path = Path(self.config['output']['final_output_directory']) / 'build_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2, default=str)

        self.logger.info(f"报告已保存: {report_path}")
        self.logger.info(f"统计信息:\n{json.dumps(self.stats, ensure_ascii=False, indent=2, default=str)}")

    def run(self):
        """执行完整流水线"""
        try:
            # 1. 加载数据
            articles = self.load_data()

            # 2. 知识抽取
            triples = self.extract_knowledge(articles)

            # 3. 导入Neo4j
            self.import_to_neo4j(triples)

            # 4. 生成报告
            self.generate_report()

            self.logger.info("=== 知识图谱构建完成 ===")

        except Exception as e:
            self.logger.error(f"流水线执行失败: {e}", exc_info=True)
            raise
        finally:
            if hasattr(self, 'neo4j_driver'):
                self.neo4j_driver.close()


def main():
    parser = argparse.ArgumentParser(description='心脏移植知识图谱自动构建')
    parser.add_argument('--config', required=True, help='配置文件路径')
    args = parser.parse_args()

    builder = AutoKGBuilder(args.config)
    builder.run()


if __name__ == '__main__':
    main()
