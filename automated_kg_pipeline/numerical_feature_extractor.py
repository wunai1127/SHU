#!/usr/bin/env python3
"""
数值特征提取器 - 将数值从文本中提取为节点属性（GNN特征）
"""

import re
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NumericalFeatureExtractor:
    """从医学文本中提取数值型特征"""

    def __init__(self):
        # 定义数值特征的正则模式
        self.patterns = {
            # 时间相关
            "ischemic_time": [
                r"缺血时间.*?(\d+\.?\d*)\s*[小时|h|hour]",
                r"ischemic\s+time.*?(\d+\.?\d*)\s*h",
            ],
            "cold_ischemic_time": [
                r"冷缺血时间.*?(\d+\.?\d*)\s*[小时|h]",
            ],

            # 血流动力学
            "pvr": [
                r"PVR.*?(\d+\.?\d*)\s*[Wood|WU]",
                r"肺血管阻力.*?(\d+\.?\d*)",
                r"pulmonary\s+vascular\s+resistance.*?(\d+\.?\d*)",
            ],
            "lvef": [
                r"LVEF.*?(\d+\.?\d*)%",
                r"左室射血分数.*?(\d+\.?\d*)%",
                r"ejection\s+fraction.*?(\d+\.?\d*)%",
            ],
            "cardiac_output": [
                r"心输出量.*?(\d+\.?\d*)\s*L/min",
            ],

            # 解剖测量
            "aortic_diameter": [
                r"主动脉.*?[直径|宽度].*?(\d+\.?\d*)\s*cm",
                r"aortic\s+diameter.*?(\d+\.?\d*)",
            ],
            "left_atrial_diameter": [
                r"左房.*?直径.*?(\d+\.?\d*)\s*cm",
            ],

            # 生化指标
            "creatinine": [
                r"肌酐.*?(\d+\.?\d*)\s*[mg/dL|umol/L]",
                r"creatinine.*?(\d+\.?\d*)",
            ],
            "bilirubin": [
                r"胆红素.*?(\d+\.?\d*)",
                r"bilirubin.*?(\d+\.?\d*)",
            ],
            "troponin": [
                r"肌钙蛋白.*?(\d+\.?\d*)",
                r"troponin.*?(\d+\.?\d*)",
            ],

            # 人口统计学
            "age": [
                r"(\d+)\s*岁",
                r"age.*?(\d+)\s*year",
                r"(\d+)[-\s]*year[-\s]*old",
            ],
            "weight": [
                r"体重.*?(\d+\.?\d*)\s*kg",
                r"weight.*?(\d+\.?\d*)\s*kg",
            ],
            "bmi": [
                r"BMI.*?(\d+\.?\d*)",
                r"体质指数.*?(\d+\.?\d*)",
            ],

            # 统计量
            "odds_ratio": [
                r"OR\s*[=:]\s*(\d+\.?\d*)",
                r"优势比\s*[=:]\s*(\d+\.?\d*)",
                r"odds\s+ratio.*?(\d+\.?\d*)",
            ],
            "p_value": [
                r"[Pp]\s*[=<]\s*(\d+\.?\d*)",
            ],
            "hazard_ratio": [
                r"HR\s*[=:]\s*(\d+\.?\d*)",
                r"风险比\s*[=:]\s*(\d+\.?\d*)",
            ],
        }

        # 单位转换
        self.unit_conversions = {
            "creatinine": {
                "umol/L_to_mg/dL": lambda x: x / 88.4,
                "mg/dL_to_umol/L": lambda x: x * 88.4,
            }
        }

    def extract_from_text(self, text: str, entity_type: Optional[str] = None) -> Dict[str, float]:
        """从文本中提取所有数值特征"""
        features = {}

        for feature_name, patterns in self.patterns.items():
            # 针对特定实体类型优化
            if entity_type:
                if not self._is_relevant_feature(feature_name, entity_type):
                    continue

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        features[feature_name] = value
                        logger.debug(f"提取特征: {feature_name} = {value} from '{text[:50]}...'")
                        break  # 找到就停止
                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析数值失败: {e}")

        return features

    def extract_from_entity(self, entity: Dict[str, Any]) -> Dict[str, float]:
        """从实体中提取数值特征"""
        features = {}

        # 从实体名称中提取
        if 'name' in entity:
            name_features = self.extract_from_text(entity['name'], entity.get('type'))
            features.update(name_features)

        # 从实体属性中提取
        if 'properties' in entity and isinstance(entity['properties'], dict):
            for key, value in entity['properties'].items():
                if isinstance(value, str):
                    prop_features = self.extract_from_text(value, entity.get('type'))
                    features.update(prop_features)

        return features

    def extract_from_relation(self, relation: Dict[str, Any]) -> Dict[str, float]:
        """从关系中提取统计量"""
        features = {}

        if 'properties' in relation:
            props = relation['properties']

            # 直接提取已结构化的统计量
            stat_fields = ['优势比', 'odds_ratio', 'OR', 'P值', 'p_value', 'HR', 'hazard_ratio']
            for field in stat_fields:
                if field in props:
                    try:
                        features[field.lower()] = float(props[field])
                    except (ValueError, TypeError):
                        pass

            # 从文本描述中提取
            if 'description' in props:
                text_features = self.extract_from_text(props['description'])
                features.update(text_features)

        return features

    def _is_relevant_feature(self, feature_name: str, entity_type: str) -> bool:
        """判断特征是否与实体类型相关"""
        relevance_map = {
            "供体特征": ["age", "ischemic_time", "cold_ischemic_time", "weight", "bmi", "aortic_diameter"],
            "受体特征": ["age", "pvr", "lvef", "creatinine", "bilirubin", "weight", "bmi"],
            "风险因子": ["ischemic_time", "pvr", "creatinine"],
            "并发症": ["troponin", "creatinine", "bilirubin"],
            "Donor": ["age", "ischemic_time", "weight", "bmi"],
            "Recipient": ["age", "pvr", "lvef", "creatinine"],
        }

        relevant_features = relevance_map.get(entity_type, [])
        return feature_name in relevant_features or not relevant_features  # 如果没定义，都接受

    def normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """归一化特征值"""
        normalized = {}

        # 定义正常范围（用于归一化）
        normal_ranges = {
            "age": (0, 100),
            "ischemic_time": (0, 8),  # 小时
            "pvr": (0, 10),  # Wood单位
            "lvef": (0, 100),  # 百分比
            "creatinine": (0, 5),  # mg/dL
            "weight": (30, 150),  # kg
            "bmi": (15, 40),
        }

        for feature_name, value in features.items():
            if feature_name in normal_ranges:
                min_val, max_val = normal_ranges[feature_name]
                # Min-Max归一化到[0, 1]
                normalized[feature_name + "_norm"] = (value - min_val) / (max_val - min_val)
                normalized[feature_name + "_norm"] = max(0, min(1, normalized[feature_name + "_norm"]))

            # 保留原始值
            normalized[feature_name] = value

        return normalized

    def create_feature_vector(self, features: Dict[str, float], feature_order: List[str]) -> List[float]:
        """创建固定顺序的特征向量（用于GNN）"""
        vector = []
        for feature_name in feature_order:
            vector.append(features.get(feature_name, 0.0))  # 缺失值填0
        return vector


# 使用示例
if __name__ == '__main__':
    extractor = NumericalFeatureExtractor()

    # 测试用例1: 从文本提取
    text1 = "缺血时间4.5小时，PVR 5.2 Wood单位，LVEF 15%"
    features1 = extractor.extract_from_text(text1)
    print(f"提取特征: {features1}")
    # 输出: {'ischemic_time': 4.5, 'pvr': 5.2, 'lvef': 15}

    # 测试用例2: 从实体提取
    entity = {
        "type": "供体特征",
        "name": "35岁男性供体",
        "properties": {
            "缺血时间": "4.5小时",
            "体重": "70kg"
        }
    }
    features2 = extractor.extract_from_entity(entity)
    print(f"实体特征: {features2}")

    # 测试用例3: 归一化
    normalized = extractor.normalize_features(features1)
    print(f"归一化特征: {normalized}")

    # 测试用例4: 特征向量
    feature_order = ['age', 'ischemic_time', 'pvr', 'lvef', 'weight']
    vector = extractor.create_feature_vector(features1, feature_order)
    print(f"特征向量: {vector}")
    # 输出: [0.0, 4.5, 5.2, 15.0, 0.0]
