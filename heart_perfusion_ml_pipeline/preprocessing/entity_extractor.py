"""
Entity Extractor - 医学实体提取模块
使用LLM进行高质量实体和关系抽取
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entity:
    """实体数据结构"""
    entity_id: str
    name: str
    entity_type: str
    aliases: List[str]
    properties: Dict[str, Any]
    source_doc_id: str
    confidence: float = 1.0


@dataclass
class Relation:
    """关系数据结构"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any]
    evidence: str
    confidence: float = 1.0


class EntityExtractor:
    """
    医学实体提取器
    结合规则和LLM进行实体关系抽取
    """

    # 实体类型定义
    ENTITY_TYPES = [
        "生理指标", "干预措施", "并发症", "器官状态", "设备参数",
        "临床指南", "药物", "检查指标", "供者特征", "受者特征",
        "风险因素", "保护因素"
    ]

    # 关系类型定义
    RELATION_TYPES = [
        "导致", "改善", "恶化", "预测",
        "监测", "干预", "禁忌", "适应",
        "参考", "包含", "相关", "对比"
    ]

    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        初始化实体提取器

        Args:
            use_llm: 是否使用LLM进行提取
            llm_client: LLM客户端
        """
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.entity_patterns = self._build_entity_patterns()
        self.relation_patterns = self._build_relation_patterns()

    def _build_entity_patterns(self) -> Dict[str, List[Tuple[str, str]]]:
        """构建实体识别的正则表达式模式"""
        return {
            "生理指标": [
                (r'乳酸(?:水平|值|浓度)?', 'lactate'),
                (r'lactate\s*(?:level|concentration)?', 'lactate'),
                (r'pH(?:值)?', 'pH'),
                (r'(?:灌注|主动脉|冠脉)压(?:力)?', 'pressure'),
                (r'(?:aortic|perfusion|coronary)\s*pressure', 'pressure'),
                (r'(?:冠脉|灌注)流量', 'flow'),
                (r'(?:coronary|perfusion)\s*flow', 'flow'),
                (r'温度', 'temperature'),
                (r'temperature', 'temperature'),
                (r'缺血时间', 'ischemic_time'),
                (r'ischemic\s*time', 'ischemic_time'),
                (r'冷缺血时间', 'cold_ischemic_time'),
                (r'cold\s*ischemic\s*time', 'cold_ischemic_time'),
                (r'(?:左室)?射血分数', 'ejection_fraction'),
                (r'(?:left\s*ventricular\s*)?ejection\s*fraction|LVEF', 'ejection_fraction'),
            ],
            "干预措施": [
                (r'加压', 'pressure_increase'),
                (r'减压', 'pressure_decrease'),
                (r'(?:机械)?灌注', 'perfusion'),
                (r'(?:mechanical\s*)?perfusion', 'perfusion'),
                (r'心肌保护液', 'cardioplegia'),
                (r'cardioplegia', 'cardioplegia'),
                (r'冷冻保存', 'cold_storage'),
                (r'cold\s*storage', 'cold_storage'),
                (r'(?:常温|低温)灌注', 'normothermic_perfusion'),
                (r'(?:normothermic|hypothermic)\s*(?:machine\s*)?perfusion', 'perfusion_type'),
            ],
            "并发症": [
                (r'原发性移植物功能障碍', 'PGD'),
                (r'primary\s*graft\s*dysfunction|PGD', 'PGD'),
                (r'(?:急性|慢性)?排斥反应', 'rejection'),
                (r'(?:acute|chronic)?\s*rejection', 'rejection'),
                (r'心力衰竭', 'heart_failure'),
                (r'heart\s*failure', 'heart_failure'),
                (r'心律失常', 'arrhythmia'),
                (r'arrhythmia', 'arrhythmia'),
                (r'缺血再灌注损伤', 'IRI'),
                (r'ischemia[-\s]*reperfusion\s*injury|IRI', 'IRI'),
            ],
            "药物": [
                (r'多巴胺', 'dopamine'),
                (r'dopamine', 'dopamine'),
                (r'(?:去甲)?肾上腺素', 'epinephrine'),
                (r'(?:nor)?epinephrine', 'epinephrine'),
                (r'环孢素', 'cyclosporine'),
                (r'cyclosporin[e]?', 'cyclosporine'),
                (r'他克莫司', 'tacrolimus'),
                (r'tacrolimus', 'tacrolimus'),
                (r'正性肌力药(?:物)?', 'inotrope'),
                (r'inotrope[s]?', 'inotrope'),
            ],
            "设备参数": [
                (r'OCS', 'OCS'),
                (r'Organ\s*Care\s*System', 'OCS'),
                (r'XVIVO', 'XVIVO'),
                (r'EVLP', 'EVLP'),
                (r'NMP', 'NMP'),
                (r'normothermic\s*machine\s*perfusion', 'NMP'),
                (r'HMP', 'HMP'),
                (r'hypothermic\s*machine\s*perfusion', 'HMP'),
            ],
        }

    def _build_relation_patterns(self) -> List[Dict[str, Any]]:
        """构建关系识别的模式"""
        return [
            # 因果关系
            {
                'pattern': r'(.+?)(?:导致|引起|造成|causes?|leads?\s*to|results?\s*in)(.+)',
                'relation': '导致',
            },
            {
                'pattern': r'(.+?)(?:改善|提高|增强|improves?|enhances?|increases?)(.+)',
                'relation': '改善',
            },
            {
                'pattern': r'(.+?)(?:恶化|降低|减少|worsens?|decreases?|reduces?)(.+)',
                'relation': '恶化',
            },
            # 预测关系
            {
                'pattern': r'(.+?)(?:预测|预示|indicates?|predicts?)(.+)',
                'relation': '预测',
            },
            # 推荐关系
            {
                'pattern': r'(?:建议|推荐|recommended?|suggested?)(.+?)(?:用于|for|when)(.+)',
                'relation': '适应',
            },
            # 禁忌关系
            {
                'pattern': r'(.+?)(?:禁忌|禁止|不建议|contraindicated?|avoid)(.+)',
                'relation': '禁忌',
            },
        ]

    def extract_entities_rule_based(self, text: str) -> List[Entity]:
        """
        基于规则的实体提取

        Args:
            text: 输入文本

        Returns:
            实体列表
        """
        entities = []
        entity_count = 0

        for entity_type, patterns in self.entity_patterns.items():
            for pattern, canonical_name in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_count += 1
                    entity = Entity(
                        entity_id=f"E{entity_count:06d}",
                        name=canonical_name,
                        entity_type=entity_type,
                        aliases=[match.group()],
                        properties={
                            'mention': match.group(),
                            'start': match.start(),
                            'end': match.end(),
                        },
                        source_doc_id="",
                        confidence=0.9
                    )
                    entities.append(entity)

        # 去重合并
        entities = self._merge_entities(entities)

        return entities

    def extract_relations_rule_based(
        self,
        text: str,
        entities: List[Entity]
    ) -> List[Relation]:
        """
        基于规则的关系提取

        Args:
            text: 输入文本
            entities: 已提取的实体列表

        Returns:
            关系列表
        """
        relations = []
        relation_count = 0

        # 构建实体位置索引
        entity_spans = []
        for e in entities:
            if 'start' in e.properties and 'end' in e.properties:
                entity_spans.append((e.properties['start'], e.properties['end'], e))

        # 按句子分割
        sentences = re.split(r'[。！？\.!\?]', text)

        for sent in sentences:
            for pattern_info in self.relation_patterns:
                pattern = pattern_info['pattern']
                relation_type = pattern_info['relation']

                match = re.search(pattern, sent, re.IGNORECASE)
                if match:
                    # 在匹配的句子中查找实体
                    sent_entities = [
                        e for e in entities
                        if e.properties.get('mention', '').lower() in sent.lower()
                    ]

                    if len(sent_entities) >= 2:
                        relation_count += 1
                        relation = Relation(
                            relation_id=f"R{relation_count:06d}",
                            source_entity_id=sent_entities[0].entity_id,
                            target_entity_id=sent_entities[1].entity_id,
                            relation_type=relation_type,
                            properties={},
                            evidence=sent.strip(),
                            confidence=0.8
                        )
                        relations.append(relation)

        return relations

    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并相同的实体"""
        merged = {}

        for e in entities:
            key = (e.name, e.entity_type)
            if key in merged:
                # 合并别名
                existing = merged[key]
                for alias in e.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
            else:
                merged[key] = e

        return list(merged.values())

    def extract_with_llm(
        self,
        text: str,
        max_length: int = 4000
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        使用LLM进行实体关系提取

        Args:
            text: 输入文本
            max_length: 最大文本长度

        Returns:
            (实体列表, 关系列表)
        """
        if not self.use_llm or not self.llm_client:
            raise ValueError("LLM client not configured")

        # 截断过长文本
        if len(text) > max_length:
            text = text[:max_length]

        prompt = self._build_extraction_prompt(text)

        # 调用LLM
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是一个医学信息抽取专家，专注于心脏移植和器官灌注领域。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)

        entities = self._parse_llm_entities(result.get('entities', []))
        relations = self._parse_llm_relations(result.get('relations', []))

        return entities, relations

    def _build_extraction_prompt(self, text: str) -> str:
        """构建LLM提取的提示词"""
        return f"""请从以下心脏移植/灌注相关文本中提取实体和关系。

文本：
{text}

请提取以下类型的实体：
{', '.join(self.ENTITY_TYPES)}

请提取以下类型的关系：
{', '.join(self.RELATION_TYPES)}

请以JSON格式返回结果：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "aliases": ["别名1", "别名2"]}}
    ],
    "relations": [
        {{"source": "源实体名称", "target": "目标实体名称", "type": "关系类型", "evidence": "支持证据"}}
    ]
}}
"""

    def _parse_llm_entities(self, llm_entities: List[Dict]) -> List[Entity]:
        """解析LLM返回的实体"""
        entities = []
        for i, e in enumerate(llm_entities):
            entity = Entity(
                entity_id=f"LLM_E{i:06d}",
                name=e.get('name', ''),
                entity_type=e.get('type', '未知'),
                aliases=e.get('aliases', []),
                properties={},
                source_doc_id="",
                confidence=0.95
            )
            entities.append(entity)
        return entities

    def _parse_llm_relations(self, llm_relations: List[Dict]) -> List[Relation]:
        """解析LLM返回的关系"""
        relations = []
        for i, r in enumerate(llm_relations):
            relation = Relation(
                relation_id=f"LLM_R{i:06d}",
                source_entity_id=r.get('source', ''),
                target_entity_id=r.get('target', ''),
                relation_type=r.get('type', '相关'),
                properties={},
                evidence=r.get('evidence', ''),
                confidence=0.9
            )
            relations.append(relation)
        return relations

    def process_document(
        self,
        doc_id: str,
        text: str,
        use_llm_for_complex: bool = True
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        处理单个文档

        Args:
            doc_id: 文档ID
            text: 文档文本
            use_llm_for_complex: 是否对复杂文档使用LLM

        Returns:
            (实体列表, 关系列表)
        """
        # 首先使用规则提取
        entities = self.extract_entities_rule_based(text)
        relations = self.extract_relations_rule_based(text, entities)

        # 设置文档ID
        for e in entities:
            e.source_doc_id = doc_id

        # 如果启用LLM且实体数量较少，使用LLM补充
        if use_llm_for_complex and self.use_llm and len(entities) < 3:
            try:
                llm_entities, llm_relations = self.extract_with_llm(text)
                for e in llm_entities:
                    e.source_doc_id = doc_id
                entities.extend(llm_entities)
                relations.extend(llm_relations)
            except Exception as e:
                print(f"LLM extraction failed: {e}")

        return entities, relations


if __name__ == "__main__":
    # 测试
    extractor = EntityExtractor(use_llm=False)

    test_text = """
    供心冷缺血时间＜8h，一般情况下心肌缺血时间<6h。
    乳酸水平升高（>4.0 mmol/L）预示灌注不足，需要增加灌注压力。
    多巴胺＜20μg·kg-1·min-1可用于维持血流动力学稳定。
    原发性移植物功能障碍（PGD）是心脏移植后主要并发症。
    """

    entities = extractor.extract_entities_rule_based(test_text)
    relations = extractor.extract_relations_rule_based(test_text, entities)

    print("Entities:")
    for e in entities:
        print(f"  {e.name} ({e.entity_type}): {e.aliases}")

    print("\nRelations:")
    for r in relations:
        print(f"  {r.source_entity_id} --{r.relation_type}--> {r.target_entity_id}")
