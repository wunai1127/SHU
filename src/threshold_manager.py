"""
阈值管理器 - 加载、查询和评估指标阈值
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class Confidence(Enum):
    """置信度等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PENDING = "pending"


class Status(Enum):
    """指标状态"""
    CONFIRMED = "confirmed"
    PROVISIONAL = "provisional"
    PENDING = "pending"


class EvaluationResult(Enum):
    """评估结果"""
    ACCEPT = "accept"
    REJECT = "reject"
    GRAY_ZONE = "gray_zone"
    RED_LINE = "red_line"
    WARNING = "warning"
    NORMAL = "normal"
    PENDING = "pending"


@dataclass
class ThresholdEvaluation:
    """阈值评估结果"""
    indicator: str
    value: float
    result: EvaluationResult
    confidence: Confidence
    message: str
    action: Optional[str] = None
    source: Optional[List[str]] = None


class ThresholdManager:
    """阈值管理器"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化阈值管理器

        Args:
            config_dir: 配置文件目录，默认为 ../config
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        self.thresholds: Dict[str, Any] = {}
        self.pending_indicators: Dict[str, Any] = {}

        self._load_configs()

    def _load_configs(self):
        """加载配置文件"""
        # 加载阈值配置
        thresholds_path = self.config_dir / "thresholds.yaml"
        if thresholds_path.exists():
            with open(thresholds_path, 'r', encoding='utf-8') as f:
                self.thresholds = yaml.safe_load(f)

        # 加载pending指标配置
        pending_path = self.config_dir / "pending_indicators.yaml"
        if pending_path.exists():
            with open(pending_path, 'r', encoding='utf-8') as f:
                self.pending_indicators = yaml.safe_load(f)

    def get_indicator_config(self, indicator: str) -> Optional[Dict[str, Any]]:
        """
        获取指标配置

        Args:
            indicator: 指标名称

        Returns:
            指标配置字典，如果不存在返回None
        """
        # 在各个类别中搜索
        for category in self.thresholds.values():
            if isinstance(category, dict) and indicator in category:
                return category[indicator]

        # 检查pending指标
        for category in self.pending_indicators.values():
            if isinstance(category, dict) and indicator in category:
                config = category[indicator]
                config['status'] = 'pending'
                return config

        return None

    def evaluate(self, indicator: str, value: float) -> ThresholdEvaluation:
        """
        评估指标值

        Args:
            indicator: 指标名称
            value: 当前值

        Returns:
            ThresholdEvaluation 评估结果
        """
        config = self.get_indicator_config(indicator)

        if config is None:
            return ThresholdEvaluation(
                indicator=indicator,
                value=value,
                result=EvaluationResult.PENDING,
                confidence=Confidence.PENDING,
                message=f"未知指标: {indicator}"
            )

        # 获取置信度
        confidence = Confidence(config.get('confidence', 'pending'))
        status = config.get('status', 'pending')
        source = config.get('source', [])

        # 如果是pending状态，仅监测
        if status == 'pending':
            return ThresholdEvaluation(
                indicator=indicator,
                value=value,
                result=EvaluationResult.PENDING,
                confidence=confidence,
                message=f"{indicator} = {value} (仅监测，无确认阈值)",
                action="monitor_only",
                source=source if isinstance(source, list) else [source]
            )

        thresholds = config.get('thresholds', {})
        result, message, action = self._evaluate_thresholds(indicator, value, thresholds)

        return ThresholdEvaluation(
            indicator=indicator,
            value=value,
            result=result,
            confidence=confidence,
            message=message,
            action=action,
            source=source if isinstance(source, list) else [source]
        )

    def _evaluate_thresholds(
        self,
        indicator: str,
        value: float,
        thresholds: Dict[str, Any]
    ) -> Tuple[EvaluationResult, str, Optional[str]]:
        """
        根据阈值配置评估值

        Returns:
            (结果, 消息, 建议操作)
        """
        # 检查红线/拒绝条件
        if 'reject' in thresholds:
            reject = thresholds['reject']
            if self._check_condition(value, reject):
                return (
                    EvaluationResult.REJECT,
                    f"{indicator} = {value} 触发拒绝条件 ({reject.get('description', '')})",
                    "拒绝/升级支持"
                )

        if 'red_line' in thresholds:
            red_line = thresholds['red_line']
            if self._check_condition(value, red_line):
                return (
                    EvaluationResult.RED_LINE,
                    f"{indicator} = {value} 触发红线 ({red_line.get('description', '')})",
                    "需要干预"
                )

        # 检查灰区
        if 'gray_zone' in thresholds:
            gz = thresholds['gray_zone']
            if gz.get('min', float('-inf')) <= value <= gz.get('max', float('inf')):
                return (
                    EvaluationResult.GRAY_ZONE,
                    f"{indicator} = {value} 处于灰区 ({gz.get('description', '')})",
                    "综合评估"
                )

        # 检查警戒
        if 'warning' in thresholds:
            warning = thresholds['warning']
            if self._check_condition(value, warning):
                return (
                    EvaluationResult.WARNING,
                    f"{indicator} = {value} 触发警戒 ({warning.get('description', '')})",
                    "加强监测/微调"
                )

        if 'warning_range' in thresholds:
            wr = thresholds['warning_range']
            if wr.get('min', float('-inf')) <= value <= wr.get('max', float('inf')):
                return (
                    EvaluationResult.WARNING,
                    f"{indicator} = {value} 处于警戒范围 ({wr.get('description', '')})",
                    "加强监测"
                )

        # 检查目标范围
        if 'target' in thresholds:
            target = thresholds['target']
            if 'min' in target and 'max' in target:
                if target['min'] <= value <= target['max']:
                    return (
                        EvaluationResult.NORMAL,
                        f"{indicator} = {value} 在目标范围内 ({target.get('description', '')})",
                        "维持当前方案"
                    )
            elif self._check_condition(value, target):
                return (
                    EvaluationResult.NORMAL,
                    f"{indicator} = {value} 达到目标 ({target.get('description', '')})",
                    "维持当前方案"
                )

        # 检查接受条件
        if 'accept' in thresholds:
            accept = thresholds['accept']
            if self._check_condition(value, accept):
                return (
                    EvaluationResult.ACCEPT,
                    f"{indicator} = {value} 满足接受条件 ({accept.get('description', '')})",
                    "接受"
                )

        # 检查操作范围
        if 'operating_range' in thresholds:
            op_range = thresholds['operating_range']
            if op_range.get('min', float('-inf')) <= value <= op_range.get('max', float('inf')):
                return (
                    EvaluationResult.NORMAL,
                    f"{indicator} = {value} 在操作范围内 ({op_range.get('description', '')})",
                    "正常运行"
                )

        # 默认返回正常
        return (
            EvaluationResult.NORMAL,
            f"{indicator} = {value}",
            None
        )

    def _check_condition(self, value: float, condition: Dict[str, Any]) -> bool:
        """检查单一条件"""
        operator = condition.get('operator')
        threshold = condition.get('value')

        if operator is None or threshold is None:
            return False

        if operator == '>':
            return value > threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<':
            return value < threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return value == threshold

        return False

    def get_all_confirmed_indicators(self) -> List[str]:
        """获取所有已确认阈值的指标列表"""
        indicators = []
        for category in self.thresholds.values():
            if isinstance(category, dict):
                for name, config in category.items():
                    if isinstance(config, dict) and config.get('status') in ['confirmed', 'provisional']:
                        indicators.append(name)
        return indicators

    def get_all_pending_indicators(self) -> List[str]:
        """获取所有pending状态的指标列表"""
        indicators = []
        for category in self.pending_indicators.values():
            if isinstance(category, dict):
                for name, config in category.items():
                    if isinstance(config, dict) and config.get('status') == 'pending':
                        indicators.append(name)
        return indicators

    def get_confidence_summary(self) -> Dict[str, List[str]]:
        """按置信度分组获取指标"""
        summary = {
            'high': [],
            'medium': [],
            'low': [],
            'pending': []
        }

        for category in self.thresholds.values():
            if isinstance(category, dict):
                for name, config in category.items():
                    if isinstance(config, dict):
                        conf = config.get('confidence', 'pending')
                        if conf in summary:
                            summary[conf].append(name)

        # 添加pending指标
        summary['pending'].extend(self.get_all_pending_indicators())

        return summary


# 便捷函数
def evaluate_indicator(indicator: str, value: float, config_dir: Optional[str] = None) -> ThresholdEvaluation:
    """快速评估单个指标"""
    manager = ThresholdManager(config_dir)
    return manager.evaluate(indicator, value)


if __name__ == "__main__":
    # 测试示例
    manager = ThresholdManager()

    print("=== 阈值评估测试 ===\n")

    # 测试已确认指标
    test_cases = [
        ("EF", 55),    # 正常
        ("EF", 42),    # 灰区
        ("EF", 35),    # 拒绝
        ("CI", 2.5),   # 正常
        ("CI", 1.8),   # 红线
        ("Lactate", 1.5),  # 理想
        ("Lactate", 6.0),  # 拒绝
        ("SvO2", 70),  # 正常
        ("SvO2", 45),  # 危急
        ("Emax", 100), # pending
    ]

    for indicator, value in test_cases:
        result = manager.evaluate(indicator, value)
        print(f"{indicator} = {value}:")
        print(f"  结果: {result.result.value}")
        print(f"  置信度: {result.confidence.value}")
        print(f"  消息: {result.message}")
        print(f"  建议: {result.action}")
        print()

    print("\n=== 置信度分组 ===")
    summary = manager.get_confidence_summary()
    for level, indicators in summary.items():
        print(f"{level}: {indicators}")
