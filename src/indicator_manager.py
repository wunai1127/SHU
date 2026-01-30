#!/usr/bin/env python3
"""
指标管理器 - 基于可调控性分类的灌注指标管理

核心设计：
1. Setpoint (可直接调控): 设备/药物可直接设定目标值
2. Readout (可间接影响): 通过调整Setpoints来改变
3. Injury Marker (只能监测): 反映损伤程度，无法直接干预

使用方法：
    manager = IndicatorManager()

    # 检查指标类型
    manager.get_indicator_type("pH")  # -> "setpoint"
    manager.get_indicator_type("Tau")  # -> "readout"

    # 获取调控建议
    manager.get_adjustment_for_readout("Tau", current_value=50)
    # -> 返回应该调整哪些Setpoints

    # 获取Setpoint目标
    manager.get_setpoint_target("Temperature")
    # -> {"initial": 22, "final": 37, "protocol": "..."}
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """指标类型"""
    SETPOINT = "setpoint"           # 可直接调控
    READOUT = "readout"             # 可间接影响
    INJURY_MARKER = "injury_marker" # 只能监测


class RiskDirection(Enum):
    """风险方向"""
    HIGHER_IS_WORSE = "higher_is_worse"
    LOWER_IS_WORSE = "lower_is_worse"


@dataclass
class SetpointIndicator:
    """可直接调控的指标"""
    name: str
    key: str
    domain: str
    unit: str
    target_range: Tuple[float, float]
    control_method: str
    device: str = ""
    typical_setpoint: Optional[float] = None
    adjustment_direction: str = ""
    monitoring_frequency: str = ""
    effects: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ReadoutIndicator:
    """不可直接调控但可间接影响的指标"""
    name: str
    key: str
    domain: str
    unit: str
    risk_threshold: float
    risk_direction: RiskDirection
    influenced_by: List[str]  # 影响它的Setpoints
    interpretation: str = ""
    notes: str = ""


@dataclass
class InjuryMarker:
    """损伤标志物"""
    name: str
    key: str
    domain: str
    unit: str
    interpretation: str
    trend_is_key: bool = True
    notes: str = ""


@dataclass
class AdjustmentRecommendation:
    """调整建议"""
    target_setpoint: str          # 建议调整的Setpoint
    setpoint_name: str            # Setpoint名称
    current_target: Tuple[float, float]  # 当前目标范围
    adjustment_direction: str     # 调整方向建议
    rationale: str               # 调整理由
    expected_effect: str         # 预期效果
    priority: int                # 优先级 (1最高)


class IndicatorManager:
    """指标管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化指标管理器

        Args:
            config_path: 配置文件路径，默认为 config/indicator_classification.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "indicator_classification.yaml"

        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path
        self.config: Dict[str, Any] = {}
        self.setpoints: Dict[str, SetpointIndicator] = {}
        self.readouts: Dict[str, ReadoutIndicator] = {}
        self.injury_markers: Dict[str, InjuryMarker] = {}
        self.causal_relationships: List[Dict] = []
        self.intervention_priority: Dict[int, str] = {}

        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}")
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 解析Setpoints
        for key, data in self.config.get("setpoints", {}).items():
            target_range = data.get("target_range", [0, 0])
            if not target_range:
                # 处理只有single target的情况
                typical = data.get("typical_setpoint", 0)
                target_range = [typical * 0.9, typical * 1.1]

            self.setpoints[key] = SetpointIndicator(
                name=data.get("name", key),
                key=key,
                domain=data.get("domain", ""),
                unit=data.get("unit", ""),
                target_range=tuple(target_range),
                control_method=data.get("control_method", ""),
                device=data.get("device", ""),
                typical_setpoint=data.get("typical_setpoint"),
                adjustment_direction=data.get("adjustment_direction", ""),
                monitoring_frequency=data.get("monitoring_frequency", ""),
                effects=data.get("effects", []),
                notes=data.get("notes", "")
            )

        # 解析Readouts
        for key, data in self.config.get("readouts", {}).items():
            risk_dir = RiskDirection.HIGHER_IS_WORSE
            if data.get("risk_direction") == "lower_is_worse":
                risk_dir = RiskDirection.LOWER_IS_WORSE

            self.readouts[key] = ReadoutIndicator(
                name=data.get("name", key),
                key=key,
                domain=data.get("domain", ""),
                unit=data.get("unit", ""),
                risk_threshold=data.get("risk_threshold", 0),
                risk_direction=risk_dir,
                influenced_by=data.get("influenced_by", []),
                interpretation=data.get("interpretation", ""),
                notes=data.get("notes", "")
            )

        # 解析Injury Markers
        for key, data in self.config.get("injury_markers", {}).items():
            self.injury_markers[key] = InjuryMarker(
                name=data.get("name", key),
                key=key,
                domain=data.get("domain", ""),
                unit=data.get("unit", ""),
                interpretation=data.get("interpretation", ""),
                trend_is_key=data.get("trend_is_key", True),
                notes=data.get("notes", "")
            )

        # 解析因果关系
        self.causal_relationships = self.config.get("causal_relationships", [])

        # 解析干预优先级
        self.intervention_priority = self.config.get("intervention_priority", {})

        logger.info(f"加载完成: {len(self.setpoints)} Setpoints, "
                    f"{len(self.readouts)} Readouts, "
                    f"{len(self.injury_markers)} Injury Markers")

    def get_indicator_type(self, key: str) -> Optional[IndicatorType]:
        """获取指标类型"""
        if key in self.setpoints:
            return IndicatorType.SETPOINT
        elif key in self.readouts:
            return IndicatorType.READOUT
        elif key in self.injury_markers:
            return IndicatorType.INJURY_MARKER
        return None

    def is_adjustable(self, key: str) -> bool:
        """判断指标是否可直接调控"""
        return key in self.setpoints

    def get_setpoint(self, key: str) -> Optional[SetpointIndicator]:
        """获取Setpoint信息"""
        return self.setpoints.get(key)

    def get_readout(self, key: str) -> Optional[ReadoutIndicator]:
        """获取Readout信息"""
        return self.readouts.get(key)

    def get_all_setpoints(self) -> Dict[str, SetpointIndicator]:
        """获取所有Setpoints"""
        return self.setpoints.copy()

    def get_all_readouts(self) -> Dict[str, ReadoutIndicator]:
        """获取所有Readouts"""
        return self.readouts.copy()

    def check_readout_risk(self, key: str, value: float) -> Tuple[bool, str]:
        """
        检查Readout是否处于风险状态

        Returns:
            (is_at_risk, risk_description)
        """
        readout = self.readouts.get(key)
        if not readout:
            return False, ""

        threshold = readout.risk_threshold
        if threshold is None:
            return False, "无阈值定义"

        if readout.risk_direction == RiskDirection.HIGHER_IS_WORSE:
            if value > threshold:
                return True, f"{readout.name}={value}{readout.unit} > {threshold} ({readout.interpretation})"
        else:
            if value < threshold:
                return True, f"{readout.name}={value}{readout.unit} < {threshold} ({readout.interpretation})"

        return False, ""

    def get_adjustment_recommendations(self, readout_key: str, current_value: float) -> List[AdjustmentRecommendation]:
        """
        获取调整建议：当某个Readout异常时，应该调整哪些Setpoints

        Args:
            readout_key: Readout的key
            current_value: 当前值

        Returns:
            调整建议列表，按优先级排序
        """
        readout = self.readouts.get(readout_key)
        if not readout:
            return []

        is_at_risk, risk_desc = self.check_readout_risk(readout_key, current_value)
        if not is_at_risk:
            return []

        recommendations = []

        # 根据influenced_by找到相关的Setpoints
        for setpoint_key in readout.influenced_by:
            setpoint = self.setpoints.get(setpoint_key)
            if not setpoint:
                continue

            # 查找因果关系
            effect_desc = ""
            for rel in self.causal_relationships:
                if rel.get("from") == setpoint_key and rel.get("to") == readout_key:
                    effect_desc = rel.get("effect", "")
                    break

            # 确定调整方向
            if readout.risk_direction == RiskDirection.HIGHER_IS_WORSE:
                # 值太高，需要降低
                direction = f"考虑调整{setpoint.name}以降低{readout.name}"
            else:
                # 值太低，需要提高
                direction = f"考虑调整{setpoint.name}以提高{readout.name}"

            # 获取优先级
            priority = 99
            for p, sp_key in self.intervention_priority.items():
                if sp_key == setpoint_key:
                    priority = p
                    break

            recommendations.append(AdjustmentRecommendation(
                target_setpoint=setpoint_key,
                setpoint_name=setpoint.name,
                current_target=setpoint.target_range,
                adjustment_direction=direction,
                rationale=f"{risk_desc}",
                expected_effect=effect_desc or f"通过{setpoint.control_method}影响{readout.name}",
                priority=priority
            ))

        # 按优先级排序
        recommendations.sort(key=lambda x: x.priority)
        return recommendations

    def get_setpoint_control_info(self, key: str) -> Dict[str, Any]:
        """获取Setpoint的控制信息（用于前端显示）"""
        setpoint = self.setpoints.get(key)
        if not setpoint:
            return {}

        return {
            "name": setpoint.name,
            "domain": setpoint.domain,
            "unit": setpoint.unit,
            "target_range": setpoint.target_range,
            "typical_setpoint": setpoint.typical_setpoint,
            "control_method": setpoint.control_method,
            "device": setpoint.device,
            "adjustment_direction": setpoint.adjustment_direction,
            "monitoring_frequency": setpoint.monitoring_frequency,
            "is_adjustable": True,
            "type": "setpoint"
        }

    def get_readout_info(self, key: str) -> Dict[str, Any]:
        """获取Readout的信息（用于前端显示）"""
        readout = self.readouts.get(key)
        if not readout:
            return {}

        return {
            "name": readout.name,
            "domain": readout.domain,
            "unit": readout.unit,
            "risk_threshold": readout.risk_threshold,
            "risk_direction": readout.risk_direction.value,
            "influenced_by": readout.influenced_by,
            "interpretation": readout.interpretation,
            "is_adjustable": False,
            "type": "readout"
        }

    def analyze_measurements(self, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        分析一组测量值

        Args:
            measurements: {指标key: 值}

        Returns:
            分析结果，包含：
            - setpoint_status: Setpoints的状态
            - readout_risks: Readouts的风险
            - recommendations: 调整建议
        """
        result = {
            "setpoint_status": [],
            "readout_risks": [],
            "recommendations": [],
            "injury_markers": []
        }

        # 检查Setpoints
        for key, value in measurements.items():
            setpoint = self.setpoints.get(key)
            if setpoint:
                low, high = setpoint.target_range
                if value < low:
                    status = "below_target"
                    message = f"{setpoint.name}={value}{setpoint.unit} 低于目标({low}-{high})"
                elif value > high:
                    status = "above_target"
                    message = f"{setpoint.name}={value}{setpoint.unit} 高于目标({low}-{high})"
                else:
                    status = "on_target"
                    message = f"{setpoint.name}={value}{setpoint.unit} 在目标范围内"

                result["setpoint_status"].append({
                    "key": key,
                    "name": setpoint.name,
                    "value": value,
                    "unit": setpoint.unit,
                    "target_range": setpoint.target_range,
                    "status": status,
                    "message": message,
                    "control_method": setpoint.control_method
                })

        # 检查Readouts
        for key, value in measurements.items():
            readout = self.readouts.get(key)
            if readout:
                is_at_risk, risk_desc = self.check_readout_risk(key, value)

                if is_at_risk:
                    result["readout_risks"].append({
                        "key": key,
                        "name": readout.name,
                        "value": value,
                        "unit": readout.unit,
                        "threshold": readout.risk_threshold,
                        "risk_direction": readout.risk_direction.value,
                        "message": risk_desc,
                        "influenced_by": readout.influenced_by
                    })

                    # 获取调整建议
                    recs = self.get_adjustment_recommendations(key, value)
                    for rec in recs:
                        result["recommendations"].append({
                            "trigger": f"{readout.name}异常",
                            "adjust_setpoint": rec.target_setpoint,
                            "setpoint_name": rec.setpoint_name,
                            "direction": rec.adjustment_direction,
                            "rationale": rec.rationale,
                            "expected_effect": rec.expected_effect,
                            "priority": rec.priority
                        })

        # 检查Injury Markers
        for key, value in measurements.items():
            marker = self.injury_markers.get(key)
            if marker:
                result["injury_markers"].append({
                    "key": key,
                    "name": marker.name,
                    "value": value,
                    "unit": marker.unit,
                    "interpretation": marker.interpretation,
                    "trend_is_key": marker.trend_is_key
                })

        # 去重并按优先级排序recommendations
        seen = set()
        unique_recs = []
        for rec in sorted(result["recommendations"], key=lambda x: x["priority"]):
            key = rec["adjust_setpoint"]
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)
        result["recommendations"] = unique_recs

        return result

    def get_consensus_causal_relationships(self) -> List[Dict]:
        """获取共识来源的因果关系（用于前端展示）"""
        consensus_rels = []
        for rel in self.causal_relationships:
            if rel.get("source") or rel.get("category"):
                consensus_rels.append(rel)
        return consensus_rels

    def get_all_indicators_summary(self) -> Dict[str, Any]:
        """获取所有指标的摘要信息（用于前端展示）"""
        return {
            "setpoint_count": len(self.setpoints),
            "readout_count": len(self.readouts),
            "injury_marker_count": len(self.injury_markers),
            "causal_count": len(self.causal_relationships),
            "consensus_causal_count": len(self.get_consensus_causal_relationships()),
            "setpoints": {k: {
                "name": v.name, "domain": v.domain, "unit": v.unit,
                "target_range": v.target_range
            } for k, v in self.setpoints.items()},
            "readouts": {k: {
                "name": v.name, "domain": v.domain, "unit": v.unit,
                "risk_threshold": v.risk_threshold,
                "risk_direction": v.risk_direction.value,
                "source": getattr(v, 'notes', '')
            } for k, v in self.readouts.items()},
            "injury_markers": {k: {
                "name": v.name, "domain": v.domain, "unit": v.unit,
                "interpretation": v.interpretation
            } for k, v in self.injury_markers.items()}
        }

    def format_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """格式化分析报告"""
        lines = ["=" * 60, "灌注指标分析报告", "=" * 60, ""]

        # Setpoint状态
        lines.append("【Setpoint 状态】(可直接调控)")
        for item in analysis["setpoint_status"]:
            icon = "✅" if item["status"] == "on_target" else "⚠️"
            lines.append(f"  {icon} {item['message']}")
            if item["status"] != "on_target":
                lines.append(f"     调控方法: {item['control_method']}")
        lines.append("")

        # Readout风险
        if analysis["readout_risks"]:
            lines.append("【Readout 风险】(需通过调整Setpoint改善)")
            for item in analysis["readout_risks"]:
                lines.append(f"  🔴 {item['message']}")
                lines.append(f"     可通过调整: {', '.join(item['influenced_by'])}")
            lines.append("")

        # 调整建议
        if analysis["recommendations"]:
            lines.append("【调整建议】(按优先级排序)")
            for i, rec in enumerate(analysis["recommendations"], 1):
                lines.append(f"  {i}. {rec['direction']}")
                lines.append(f"     原因: {rec['rationale']}")
                lines.append(f"     预期: {rec['expected_effect']}")
            lines.append("")

        # Injury Markers
        if analysis["injury_markers"]:
            lines.append("【损伤标志物】(仅供监测)")
            for item in analysis["injury_markers"]:
                lines.append(f"  📊 {item['name']}: {item['value']} {item['unit']}")
                lines.append(f"     解读: {item['interpretation']}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# 命令行测试
# =============================================================================
if __name__ == "__main__":
    manager = IndicatorManager()

    print("\n所有 Setpoints (可直接调控):")
    print("-" * 40)
    for key, sp in manager.get_all_setpoints().items():
        print(f"  {key}: {sp.name} ({sp.domain})")
        print(f"    目标: {sp.target_range} {sp.unit}")
        print(f"    调控: {sp.control_method}")
        print()

    print("\n所有 Readouts (可间接影响):")
    print("-" * 40)
    for key, rd in manager.get_all_readouts().items():
        print(f"  {key}: {rd.name} ({rd.domain})")
        dir_str = "越高越差" if rd.risk_direction == RiskDirection.HIGHER_IS_WORSE else "越低越差"
        print(f"    阈值: {rd.risk_threshold} {rd.unit} ({dir_str})")
        print(f"    受影响于: {rd.influenced_by}")
        print()

    # 测试分析
    print("\n" + "=" * 60)
    print("测试分析")
    print("=" * 60)

    test_measurements = {
        "pH": 7.30,
        "Temperature": 35,
        "Dobutamine": 4,
        "Tau": 50,           # > 44 风险
        "EF": 15,            # < 18 风险
        "CVR": 0.045,        # > 0.040 风险
        "Lactate_arterial": 2.0  # > 1.55 风险
    }

    analysis = manager.analyze_measurements(test_measurements)
    report = manager.format_analysis_report(analysis)
    print(report)
