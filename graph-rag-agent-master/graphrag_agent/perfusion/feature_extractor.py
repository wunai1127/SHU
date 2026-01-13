"""
Heart Anatomy Feature Extractor

从心脏评估文本中提取数值特征

Extracts numerical features from free-text heart assessment reports:
- Anatomical measurements (aortic diameter, wall thickness, etc.)
- Functional assessments (ejection fraction, contractility)
- Risk factors (calcification, stenosis, etc.)

Usage:
    extractor = HeartAnatomyExtractor()
    features = extractor.extract("主动脉根部直径3.2cm，左心室壁厚度1.1cm")
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of feature extraction"""
    numerical_features: Dict[str, float]
    categorical_features: Dict[str, str]
    risk_factors: List[str]
    raw_text: str
    confidence: float = 1.0
    extraction_method: str = "regex"  # "regex", "llm", "hybrid"


class HeartAnatomyExtractor:
    """
    Extract numerical and categorical features from heart assessment text

    Supports both English and Chinese text patterns
    """

    # Regex patterns for numerical features
    # 数值特征的正则表达式模式（支持中英文）
    NUMERICAL_PATTERNS = {
        'aortic_diameter_cm': [
            # English patterns
            r'aort(?:ic|a).*?(?:root\s+)?(?:diameter|width).*?(\d+\.?\d*)\s*(?:cm|mm)',
            r'(\d+\.?\d*)\s*(?:cm|mm).*?aort(?:ic|a)',
            r'aortic\s+root[:\s]+(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese patterns
            r'主动脉(?:根部)?(?:直径|宽度).*?(\d+\.?\d*)\s*(?:cm|厘米|mm|毫米)',
            r'(\d+\.?\d*)\s*(?:cm|厘米).*?主动脉',
        ],
        'lv_wall_thickness_cm': [
            r'(?:left\s+)?ventricle.*?(?:wall\s+)?thickness.*?(\d+\.?\d*)\s*(?:cm|mm)',
            r'LV\s*(?:wall\s+)?(?:thickness)?.*?(\d+\.?\d*)\s*(?:cm|mm)',
            r'IVS[d]?\s*[:=]?\s*(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese
            r'左心室壁厚度.*?(\d+\.?\d*)\s*(?:cm|厘米|mm)',
            r'室间隔厚度.*?(\d+\.?\d*)\s*(?:cm|厘米|mm)',
        ],
        'ejection_fraction_percent': [
            r'(?:ejection\s+fraction|EF|LVEF)[:\s]*(\d+\.?\d*)%?',
            r'EF\s*[:=]?\s*(\d+\.?\d*)%?',
            # Chinese
            r'射血分数.*?(\d+\.?\d*)%?',
            r'EF值?.*?(\d+\.?\d*)%?',
        ],
        'lv_end_diastolic_diameter_cm': [
            r'LV(?:ID)?[dD]\s*[:=]?\s*(\d+\.?\d*)\s*(?:cm|mm)',
            r'(?:left\s+ventricle|LV).*?(?:end[- ]?diastolic|diastolic).*?(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese
            r'左室舒张末(?:内)?径.*?(\d+\.?\d*)\s*(?:cm|mm)',
        ],
        'lv_end_systolic_diameter_cm': [
            r'LV(?:ID)?[sS]\s*[:=]?\s*(\d+\.?\d*)\s*(?:cm|mm)',
            r'(?:left\s+ventricle|LV).*?(?:end[- ]?systolic|systolic).*?(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese
            r'左室收缩末(?:内)?径.*?(\d+\.?\d*)\s*(?:cm|mm)',
        ],
        'la_diameter_cm': [
            r'(?:left\s+atrium|LA)\s*[:=]?\s*(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese
            r'左房(?:内)?径.*?(\d+\.?\d*)\s*(?:cm|mm)',
        ],
        'rv_diameter_cm': [
            r'(?:right\s+ventricle|RV)\s*[:=]?\s*(\d+\.?\d*)\s*(?:cm|mm)',
            # Chinese
            r'右室(?:内)?径.*?(\d+\.?\d*)\s*(?:cm|mm)',
        ],
        'pa_pressure_mmhg': [
            r'(?:pulmonary\s+artery|PA)\s+pressure.*?(\d+\.?\d*)\s*(?:mmHg)?',
            r'PASP\s*[:=]?\s*(\d+\.?\d*)',
            # Chinese
            r'肺动脉压.*?(\d+\.?\d*)\s*(?:mmHg)?',
        ],
        'heart_rate_bpm': [
            r'(?:heart\s+rate|HR)\s*[:=]?\s*(\d+)\s*(?:bpm)?',
            # Chinese
            r'心率.*?(\d+)\s*(?:次/分|bpm)?',
        ],
        'ischemic_time_minutes': [
            r'(?:cold\s+)?ischemic\s+time.*?(\d+\.?\d*)\s*(?:min|minutes|h|hours)',
            # Chinese
            r'(?:冷)?缺血时间.*?(\d+\.?\d*)\s*(?:分钟|小时|min|h)',
        ],
        'donor_age_years': [
            r'(?:donor\s+)?age[:\s]+(\d+)\s*(?:years|yo|y)?',
            # Chinese
            r'(?:供体)?年龄.*?(\d+)\s*岁?',
        ],
    }

    # Categorical feature patterns
    # 分类特征模式
    CATEGORICAL_PATTERNS = {
        'valve_condition': {
            'patterns': [
                (r'(?:mitral|aortic|tricuspid|pulmonary)\s+(?:valve\s+)?(?:severe|significant)\s+(?:regurgitation|stenosis|insufficiency)', 'severe'),
                (r'(?:mitral|aortic|tricuspid|pulmonary)\s+(?:valve\s+)?(?:moderate)\s+(?:regurgitation|stenosis)', 'moderate'),
                (r'(?:mitral|aortic|tricuspid|pulmonary)\s+(?:valve\s+)?(?:mild|trace|trivial)\s+(?:regurgitation|stenosis)', 'mild'),
                (r'(?:valve|valves).*?(?:normal|intact|no\s+significant)', 'normal'),
                # Chinese
                (r'(?:二尖瓣|主动脉瓣|三尖瓣).*?(?:重度|严重).*?(?:反流|狭窄|关闭不全)', 'severe'),
                (r'(?:二尖瓣|主动脉瓣|三尖瓣).*?(?:中度).*?(?:反流|狭窄)', 'moderate'),
                (r'(?:二尖瓣|主动脉瓣|三尖瓣).*?(?:轻度|轻微).*?(?:反流|狭窄)', 'mild'),
                (r'瓣膜.*?(?:正常|未见异常)', 'normal'),
            ],
            'default': 'unknown'
        },
        'coronary_anatomy': {
            'patterns': [
                (r'(?:coronary|LAD|LCX|RCA).*?(?:severe|significant|critical)\s+(?:stenosis|occlusion|disease)', 'severe_disease'),
                (r'(?:coronary|LAD|LCX|RCA).*?(?:moderate)\s+(?:stenosis|disease)', 'moderate_disease'),
                (r'(?:coronary|LAD|LCX|RCA).*?(?:mild|minor)\s+(?:stenosis|disease|plaque)', 'mild_disease'),
                (r'(?:coronary|coronaries).*?(?:calcif|plaque)', 'calcification'),
                (r'(?:coronary|coronaries).*?(?:normal|patent|no\s+significant)', 'normal'),
                # Chinese
                (r'冠状?动脉.*?(?:重度|严重|显著).*?(?:狭窄|闭塞)', 'severe_disease'),
                (r'冠状?动脉.*?(?:中度).*?狭窄', 'moderate_disease'),
                (r'冠状?动脉.*?(?:轻度|轻微).*?(?:狭窄|斑块)', 'mild_disease'),
                (r'冠状?动脉.*?(?:钙化|斑块)', 'calcification'),
                (r'冠状?动脉.*?(?:通畅|正常|未见狭窄)', 'normal'),
            ],
            'default': 'unknown'
        },
        'contractility': {
            'patterns': [
                (r'(?:contractility|contraction|systolic\s+function).*?(?:excellent|hyperdynamic|good)', 'excellent'),
                (r'(?:contractility|contraction|systolic\s+function).*?(?:normal|preserved)', 'good'),
                (r'(?:contractility|contraction|systolic\s+function).*?(?:mild(?:ly)?\s+reduced|fair)', 'fair'),
                (r'(?:contractility|contraction|systolic\s+function).*?(?:moderate(?:ly)?\s+reduced)', 'reduced'),
                (r'(?:contractility|contraction|systolic\s+function).*?(?:severe(?:ly)?\s+reduced|poor)', 'poor'),
                # Chinese
                (r'收缩(?:功能|力).*?(?:良好|正常)', 'good'),
                (r'收缩(?:功能|力).*?(?:轻度减低)', 'fair'),
                (r'收缩(?:功能|力).*?(?:中度减低)', 'reduced'),
                (r'收缩(?:功能|力).*?(?:重度减低|差)', 'poor'),
            ],
            'default': 'unknown'
        },
        'rhythm': {
            'patterns': [
                (r'(?:sinus|normal)\s+rhythm', 'sinus'),
                (r'(?:atrial\s+fibrillation|AF|afib)', 'afib'),
                (r'(?:atrial\s+flutter)', 'aflutter'),
                (r'(?:ventricular\s+tachycardia|VT)', 'vtach'),
                (r'(?:paced|pacemaker)', 'paced'),
                # Chinese
                (r'窦性心律', 'sinus'),
                (r'心房颤动|房颤', 'afib'),
                (r'心房扑动|房扑', 'aflutter'),
                (r'室性心动过速', 'vtach'),
            ],
            'default': 'unknown'
        },
        'donor_type': {
            'patterns': [
                (r'\bDBD\b|brain\s*dead|brain\s*death', 'DBD'),
                (r'\bDCD\b|(?:donation|donor)\s+after\s+(?:circulatory|cardiac)\s+death', 'DCD'),
                # Chinese
                (r'脑死亡', 'DBD'),
                (r'心脏死亡|DCD', 'DCD'),
            ],
            'default': 'unknown'
        }
    }

    # Risk factor patterns
    # 风险因素模式
    RISK_FACTOR_PATTERNS = [
        (r'(?:severe|significant)\s+(?:LVH|left\s+ventricular\s+hypertrophy)', 'severe_lvh'),
        (r'(?:prior|previous)\s+(?:MI|myocardial\s+infarction|heart\s+attack)', 'prior_mi'),
        (r'(?:dilated|enlarged)\s+(?:heart|ventricle|cardiomyopathy)', 'dilated_cardiomyopathy'),
        (r'(?:wall\s+motion|regional)\s+abnormal', 'wall_motion_abnormality'),
        (r'pericardial\s+effusion', 'pericardial_effusion'),
        (r'(?:inotropic|vasopressor)\s+support', 'inotropic_support'),
        (r'prolonged\s+(?:ischemic|ischemia)\s+time', 'prolonged_ischemia'),
        (r'(?:cardiac|cardiopulmonary)\s+arrest', 'cardiac_arrest_history'),
        (r'(?:sepsis|septic)', 'sepsis'),
        (r'(?:diabetes|diabetic)', 'diabetes'),
        (r'(?:hypertension|hypertensive)', 'hypertension'),
        # Chinese
        (r'(?:重度|显著)\s*(?:左室肥厚|LVH)', 'severe_lvh'),
        (r'(?:既往|陈旧性)\s*心肌梗死', 'prior_mi'),
        (r'(?:扩张型)?心肌病', 'dilated_cardiomyopathy'),
        (r'室壁运动异常', 'wall_motion_abnormality'),
        (r'心包积液', 'pericardial_effusion'),
        (r'(?:正性|血管活性).*?药物', 'inotropic_support'),
        (r'缺血时间(?:延长|过长)', 'prolonged_ischemia'),
        (r'心脏骤停', 'cardiac_arrest_history'),
        (r'脓毒症|感染性休克', 'sepsis'),
        (r'糖尿病', 'diabetes'),
        (r'高血压', 'hypertension'),
    ]

    def __init__(self, llm_client: Any = None):
        """
        Initialize extractor

        Args:
            llm_client: Optional LLM client for fallback extraction
        """
        self.llm = llm_client

    def extract(self, text: str) -> ExtractionResult:
        """
        Extract features from text

        Args:
            text: Heart assessment text (English or Chinese)

        Returns:
            ExtractionResult with numerical, categorical features and risk factors
        """
        text_lower = text.lower()
        numerical = {}
        categorical = {}
        risk_factors = []

        # Extract numerical features
        for feature_name, patterns in self.NUMERICAL_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        # Convert mm to cm if needed
                        if 'mm' in pattern.lower() and value > 10:
                            value = value / 10
                        numerical[feature_name] = value
                        break
                    except ValueError:
                        continue

        # Extract categorical features
        for feature_name, config in self.CATEGORICAL_PATTERNS.items():
            for pattern, value in config['patterns']:
                if re.search(pattern, text, re.IGNORECASE):
                    categorical[feature_name] = value
                    break
            if feature_name not in categorical:
                categorical[feature_name] = config['default']

        # Extract risk factors
        for pattern, factor in self.RISK_FACTOR_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                risk_factors.append(factor)

        return ExtractionResult(
            numerical_features=numerical,
            categorical_features=categorical,
            risk_factors=list(set(risk_factors)),  # Remove duplicates
            raw_text=text,
            confidence=self._calculate_confidence(numerical, categorical),
            extraction_method="regex"
        )

    def extract_with_llm_fallback(self, text: str) -> ExtractionResult:
        """
        Extract features using regex first, then LLM for missing features

        先用正则提取，缺失特征用LLM兜底
        """
        # Try regex first
        result = self.extract(text)

        # Check if key features are missing
        key_features = ['ejection_fraction_percent', 'aortic_diameter_cm']
        missing = [f for f in key_features if f not in result.numerical_features]

        if missing and self.llm:
            try:
                llm_result = self._extract_with_llm(text, missing)
                result.numerical_features.update(llm_result.get('numerical', {}))
                result.categorical_features.update(llm_result.get('categorical', {}))
                result.risk_factors.extend(llm_result.get('risk_factors', []))
                result.extraction_method = "hybrid"
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")

        return result

    def _extract_with_llm(self, text: str, missing_features: List[str]) -> Dict:
        """Use LLM to extract missing features"""
        if not self.llm:
            return {}

        prompt = f"""Extract the following medical measurements from this heart assessment text:
Missing features needed: {missing_features}

Text:
{text}

Return a JSON object with:
{{
    "numerical": {{"feature_name": value, ...}},
    "categorical": {{"feature_name": "value", ...}},
    "risk_factors": ["factor1", "factor2", ...]
}}

Only include features you can confidently extract. Use null for uncertain values.
"""

        try:
            response = self.llm.invoke(prompt)
            import json
            return json.loads(response.content)
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}")
            return {}

    def _calculate_confidence(
        self,
        numerical: Dict[str, float],
        categorical: Dict[str, str]
    ) -> float:
        """
        Calculate extraction confidence based on features found

        根据提取到的特征数量计算置信度
        """
        # Key features that should be present for high confidence
        key_numerical = ['ejection_fraction_percent', 'aortic_diameter_cm', 'lv_wall_thickness_cm']
        key_categorical = ['valve_condition', 'contractility', 'coronary_anatomy']

        numerical_found = sum(1 for k in key_numerical if k in numerical)
        categorical_found = sum(1 for k in key_categorical
                               if k in categorical and categorical[k] != 'unknown')

        total_key = len(key_numerical) + len(key_categorical)
        found = numerical_found + categorical_found

        return found / total_key if total_key > 0 else 0.5

    def to_feature_vector(
        self,
        result: ExtractionResult,
        include_categorical: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Convert extraction result to numerical feature vector

        将提取结果转换为数值特征向量，用于GNN输入

        Args:
            result: Extraction result
            include_categorical: Whether to one-hot encode categorical features

        Returns:
            Tuple of (feature_vector, feature_names)
        """
        import numpy as np

        features = []
        names = []

        # Numerical features with defaults
        numerical_defaults = {
            'aortic_diameter_cm': 3.0,
            'lv_wall_thickness_cm': 1.0,
            'ejection_fraction_percent': 55.0,
            'lv_end_diastolic_diameter_cm': 5.0,
            'lv_end_systolic_diameter_cm': 3.5,
            'la_diameter_cm': 4.0,
            'heart_rate_bpm': 70,
            'donor_age_years': 40,
            'ischemic_time_minutes': 240,
        }

        for name, default in numerical_defaults.items():
            value = result.numerical_features.get(name, default)
            features.append(value)
            names.append(name)

        if include_categorical:
            # One-hot encode categorical features
            categorical_encodings = {
                'valve_condition': ['normal', 'mild', 'moderate', 'severe', 'unknown'],
                'coronary_anatomy': ['normal', 'calcification', 'mild_disease', 'moderate_disease', 'severe_disease', 'unknown'],
                'contractility': ['excellent', 'good', 'fair', 'reduced', 'poor', 'unknown'],
                'rhythm': ['sinus', 'afib', 'aflutter', 'vtach', 'paced', 'unknown'],
                'donor_type': ['DBD', 'DCD', 'unknown'],
            }

            for cat_name, options in categorical_encodings.items():
                value = result.categorical_features.get(cat_name, 'unknown')
                for option in options:
                    features.append(1.0 if value == option else 0.0)
                    names.append(f"{cat_name}_{option}")

            # Risk factors as binary features
            all_risk_factors = [
                'severe_lvh', 'prior_mi', 'dilated_cardiomyopathy',
                'wall_motion_abnormality', 'pericardial_effusion',
                'inotropic_support', 'prolonged_ischemia',
                'cardiac_arrest_history', 'sepsis', 'diabetes', 'hypertension'
            ]

            for rf in all_risk_factors:
                features.append(1.0 if rf in result.risk_factors else 0.0)
                names.append(f"risk_{rf}")

        return np.array(features, dtype=np.float32), names


# Import numpy for to_feature_vector
import numpy as np


if __name__ == "__main__":
    # Test the extractor
    print("Testing HeartAnatomyExtractor...")

    extractor = HeartAnatomyExtractor()

    # Test with English text
    english_text = """
    Heart Assessment Report:
    - Aortic root diameter: 3.2 cm, no significant stenosis
    - Left ventricle wall thickness: 1.1 cm, systolic function good
    - Ejection fraction (EF): 55%
    - Mild mitral regurgitation, aortic valve normal
    - Coronary arteries: Left anterior descending with calcified plaque, no significant stenosis
    - Heart rate: 72 bpm, sinus rhythm
    - Donor type: DBD, age 45 years
    """

    result = extractor.extract(english_text)
    print("\n=== English Text Results ===")
    print(f"Numerical features: {result.numerical_features}")
    print(f"Categorical features: {result.categorical_features}")
    print(f"Risk factors: {result.risk_factors}")
    print(f"Confidence: {result.confidence:.2f}")

    # Test with Chinese text
    chinese_text = """
    供心评估报告：
    - 主动脉根部直径3.2cm，未见明显狭窄
    - 左心室壁厚度1.1cm，收缩功能良好
    - 射血分数EF：55%
    - 二尖瓣轻度反流，主动脉瓣功能正常
    - 冠状动脉左前降支近端钙化斑块，无明显狭窄
    - 心率72次/分，窦性心律
    - 供体类型：脑死亡，年龄45岁
    - 冷缺血时间：180分钟
    """

    result_cn = extractor.extract(chinese_text)
    print("\n=== Chinese Text Results ===")
    print(f"Numerical features: {result_cn.numerical_features}")
    print(f"Categorical features: {result_cn.categorical_features}")
    print(f"Risk factors: {result_cn.risk_factors}")
    print(f"Confidence: {result_cn.confidence:.2f}")

    # Test feature vector conversion
    vector, names = extractor.to_feature_vector(result)
    print(f"\n=== Feature Vector ===")
    print(f"Vector shape: {vector.shape}")
    print(f"Feature names: {names[:10]}...")  # First 10

    print("\nAll tests passed!")
