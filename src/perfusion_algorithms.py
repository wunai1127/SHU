#!/usr/bin/env python3
"""
灌注监测算法模块 — 四大核心算法
=================================

1. SRCO (Setpoint-Readout Causal Optimization)
   基于因果图的Setpoint调控优化 — 当Readout异常时，反向推理最优Setpoint调整方案

2. CUSUM Early Warning
   累积和控制图 — 在指标突破阈值前检测恶化趋势，提前预警

3. CPRS (Composite Perfusion Risk Score)
   复合灌注风险评分 — 融合所有指标的加权风险分数，单一数值反映灌注全局质量

4. Lactate Trajectory Prediction
   乳酸清除轨迹预测 — 拟合指数衰减模型，预测乳酸清除率和达标时间
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# 1. SRCO — Setpoint-Readout Causal Optimization
# =============================================================================

class SetpointReadoutCausalOptimizer:
    """
    基于因果图的Setpoint调控优化算法

    核心思想:
    - 构建 Setpoint→Readout 的有向因果图（权重 = 影响强度）
    - 当Readout异常时，反向遍历因果图找到可调控的Setpoint
    - 通过影响传播权重排序，推荐最有效的Setpoint调整方案
    - 综合考虑: (1)Readout偏离程度 (2)因果影响权重 (3)Setpoint调控优先级

    算法复杂度: O(S×R) 其中 S=Setpoint数, R=Readout数
    """

    # Setpoint → Readout 因果邻接矩阵（权重0-1，代表影响强度）
    # 来源: indicator_classification.yaml 的 influenced_by 和 causal_relationships
    CAUSAL_EDGES = {
        # Temperature 影响最广
        ("Temperature", "CVR"):      {"weight": 0.9, "direction": "negative", "mechanism": "低温→CVR↑"},
        ("Temperature", "Tau"):      {"weight": 0.8, "direction": "negative", "mechanism": "温度↑→舒张改善→Tau↓"},
        ("Temperature", "SW"):       {"weight": 0.5, "direction": "positive", "mechanism": "温度正常化→SW恢复"},
        ("Temperature", "MVO2"):     {"weight": 0.6, "direction": "positive", "mechanism": "温度↑→代谢率↑→MVO2↑"},
        ("Temperature", "Lactate"):  {"weight": 0.5, "direction": "negative", "mechanism": "温度正常化→代谢改善→Lactate↓"},
        ("Temperature", "PVR"):      {"weight": 0.4, "direction": "negative", "mechanism": "温度↑→PVR↓"},
        ("Temperature", "dPdt_min"): {"weight": 0.5, "direction": "negative", "mechanism": "温度↑→舒张↑→|dPdt_min|↑"},
        ("Temperature", "O2_ext"):   {"weight": 0.3, "direction": "negative", "mechanism": "温度正常化→氧代谢优化"},

        # AoDP 灌注压基础
        ("AoDP", "DevP"):           {"weight": 0.9, "direction": "positive", "mechanism": "AoDP→冠脉灌注→DevP"},
        ("AoDP", "EF"):             {"weight": 0.7, "direction": "positive", "mechanism": "AoDP目标→冠脉灌注→EF维持"},
        ("AoDP", "SW"):             {"weight": 0.6, "direction": "positive", "mechanism": "AoDP→后负荷稳定→SW"},
        ("AoDP", "CVR"):            {"weight": 0.5, "direction": "positive", "mechanism": "AoDP维持→CVR稳定"},
        ("AoDP", "GFR"):            {"weight": 0.6, "direction": "positive", "mechanism": "AoDP→肾灌注→GFR"},
        ("AoDP", "Creatinine"):     {"weight": 0.5, "direction": "negative", "mechanism": "AoDP↑→肾灌注改善→Cr↓"},

        # Dobutamine 正性肌力（影响最多readouts）
        ("Dobutamine", "dPdt_max"): {"weight": 0.95, "direction": "positive", "mechanism": "Dobutamine→收缩力↑→dPdt_max↑"},
        ("Dobutamine", "EF"):       {"weight": 0.85, "direction": "positive", "mechanism": "Dobutamine→EF↑"},
        ("Dobutamine", "SW"):       {"weight": 0.7, "direction": "positive", "mechanism": "Dobutamine→心肌做功↑→SW↑"},
        ("Dobutamine", "dPdt_min"): {"weight": 0.6, "direction": "positive", "mechanism": "Dobutamine→lusitropy↑→|dPdt_min|↑"},
        ("Dobutamine", "Tau"):      {"weight": 0.5, "direction": "negative", "mechanism": "Dobutamine→舒张改善→Tau↓"},
        ("Dobutamine", "MVO2"):     {"weight": 0.7, "direction": "positive", "mechanism": "Dobutamine→氧耗↑→MVO2↑"},
        ("Dobutamine", "O2_ext"):   {"weight": 0.4, "direction": "positive", "mechanism": "Dobutamine→耗氧↑→O2提取↑"},
        ("Dobutamine", "CI"):       {"weight": 0.8, "direction": "positive", "mechanism": "Dobutamine→CO↑→CI↑"},
        ("Dobutamine", "GFR"):      {"weight": 0.4, "direction": "positive", "mechanism": "Dobutamine→CO↑→肾灌注↑→GFR↑"},

        # LAP 前负荷
        ("LAP", "SW"):              {"weight": 0.8, "direction": "positive", "mechanism": "LAP↑→前负荷↑→SW↑(Frank-Starling)"},
        ("LAP", "EF"):              {"weight": 0.6, "direction": "positive", "mechanism": "LAP→前负荷优化→EF"},
        ("LAP", "Tau"):             {"weight": 0.5, "direction": "positive", "mechanism": "LAP↑→舒张负荷↑→Tau可能↑"},
        ("LAP", "dPdt_min"):        {"weight": 0.4, "direction": "positive", "mechanism": "LAP→前负荷→舒张功能"},
        ("LAP", "MVO2"):            {"weight": 0.5, "direction": "positive", "mechanism": "LAP↑→做功↑→MVO2↑"},

        # PaO2 氧供
        ("PaO2", "Lactate"):        {"weight": 0.8, "direction": "negative", "mechanism": "PaO2↑→氧供↑→Lactate↓"},
        ("PaO2", "O2_ext"):         {"weight": 0.6, "direction": "negative", "mechanism": "PaO2↑→CaO2↑→O2提取率↓"},
        ("PaO2", "PVR"):            {"weight": 0.5, "direction": "negative", "mechanism": "PaO2↑→肺血管扩张→PVR↓"},
        ("PaO2", "SvO2"):           {"weight": 0.5, "direction": "positive", "mechanism": "PaO2↑→氧供↑→SvO2↑"},

        # Hemoglobin 携氧
        ("Hemoglobin", "MVO2"):     {"weight": 0.8, "direction": "positive", "mechanism": "Hb↑→CaO2↑→MVO2↑"},
        ("Hemoglobin", "CVR"):      {"weight": 0.4, "direction": "positive", "mechanism": "Hb→血粘滞度→CVR"},
        ("Hemoglobin", "O2_ext"):   {"weight": 0.6, "direction": "negative", "mechanism": "Hb↑→CaO2↑→O2提取率↓"},
        ("Hemoglobin", "Lactate"):  {"weight": 0.6, "direction": "negative", "mechanism": "Hb↑→DO2↑→Lactate↓"},
        ("Hemoglobin", "SvO2"):     {"weight": 0.5, "direction": "positive", "mechanism": "Hb↑→DO2↑→SvO2↑"},

        # PacingRate 起搏
        ("PacingRate", "Tau"):      {"weight": 0.6, "direction": "positive", "mechanism": "HR↑→舒张时间↓→Tau↑"},
        ("PacingRate", "EF"):       {"weight": 0.4, "direction": "complex", "mechanism": "HR→Bowditch效应(适度)→EF变化"},
        ("PacingRate", "dPdt_max"): {"weight": 0.5, "direction": "positive", "mechanism": "HR↑→Bowditch→dPdt_max↑"},
        ("PacingRate", "dPdt_min"): {"weight": 0.5, "direction": "positive", "mechanism": "HR↑→舒张变化→dPdt_min"},

        # Insulin 代谢支持（间接影响）
        ("Insulin", "Lactate"):     {"weight": 0.3, "direction": "negative", "mechanism": "胰岛素→底物利用优化→Lactate↓"},

        # Flow 灌注流量（最核心，间接通过AoDP、DO2影响）
        ("Flow", "Lactate"):        {"weight": 0.9, "direction": "negative", "mechanism": "Flow↑→DO2↑→Lactate↓"},
        ("Flow", "MVO2"):           {"weight": 0.7, "direction": "positive", "mechanism": "Flow↑→DO2↑→MVO2↑"},
        ("Flow", "SvO2"):           {"weight": 0.6, "direction": "positive", "mechanism": "Flow↑→DO2↑→SvO2↑"},
        ("Flow", "GFR"):            {"weight": 0.5, "direction": "positive", "mechanism": "Flow↑→肾灌注↑→GFR↑"},
    }

    # Setpoint 调控优先级（越小越优先调整）
    SETPOINT_PRIORITY = {
        "Temperature": 1,
        "AoDP": 2,
        "Flow": 2,  # 与AoDP同等优先
        "Dobutamine": 3,
        "LAP": 4,
        "PaO2": 5,
        "Hemoglobin": 6,
        "PacingRate": 7,
        "pH": 8,
        "Insulin": 9,
    }

    # Readout 目标范围和异常方向
    READOUT_TARGETS = {
        "Lactate":    {"target_mid": 2.0,  "range": (0, 4.0),  "higher_worse": True},
        "pH":         {"target_mid": 7.30, "range": (7.25, 7.35), "higher_worse": False},
        "K_A":        {"target_mid": 4.25, "range": (3.5, 5.0), "higher_worse": True},
        "EF":         {"target_mid": 35,   "range": (18, 60),   "higher_worse": False},
        "CI":         {"target_mid": 3.0,  "range": (2.2, 4.0), "higher_worse": False},
        "SvO2":       {"target_mid": 72,   "range": (65, 80),   "higher_worse": False},
        "CvO2":       {"target_mid": 14,   "range": (12, 16),   "higher_worse": False},
        "MVO2":       {"target_mid": 12,   "range": (8.8, 20),  "higher_worse": False},
        "dPdt_max":   {"target_mid": 1500, "range": (1200, 1800), "higher_worse": False},
        "CVR":        {"target_mid": 0.03, "range": (0.01, 0.04), "higher_worse": True},
        "Tau":        {"target_mid": 37,   "range": (30, 44),   "higher_worse": True},
        "SW":         {"target_mid": 1500, "range": (1309, 2000), "higher_worse": False},
        "O2_ext":     {"target_mid": 12,   "range": (5, 17),    "higher_worse": True},
        "PVR":        {"target_mid": 1.5,  "range": (0.5, 2.5), "higher_worse": True},
        "GFR":        {"target_mid": 80,   "range": (60, 120),  "higher_worse": False},
        "Creatinine": {"target_mid": 1.0,  "range": (0.5, 1.5), "higher_worse": True},
    }

    @classmethod
    def compute_readout_deviation(cls, readout: str, value: float) -> float:
        """
        计算Readout偏离度 ∈ [0, 1]

        公式: deviation = |value - target_mid| / normalization_range
        归一化使不同量纲的指标可比
        """
        cfg = cls.READOUT_TARGETS.get(readout)
        if not cfg:
            return 0.0
        lo, hi = cfg["range"]
        mid = cfg["target_mid"]
        span = hi - lo if hi != lo else 1.0

        if lo <= value <= hi:
            return abs(value - mid) / span  # 在范围内，偏离中心的程度
        elif value > hi:
            return min(1.0, (value - mid) / span)
        else:
            return min(1.0, (mid - value) / span)

    @classmethod
    def optimize(cls, current_readouts: Dict[str, float],
                 current_setpoints: Dict[str, float] = None) -> List[Dict]:
        """
        SRCO核心算法: 给定当前Readout值，推荐最优Setpoint调整方案

        算法流程:
        1. 计算每个Readout的偏离度 d(r)
        2. 对每个异常Readout，反向遍历因果图找到影响它的Setpoints
        3. 计算每个Setpoint的综合调控收益:
           Benefit(s) = Σ_r [ d(r) × w(s→r) × (1 / priority(s)) ]
        4. 按Benefit降序排列，输出调控建议

        Returns:
            [{
                "setpoint": str,
                "benefit_score": float,  # 调控收益分
                "priority": int,
                "affected_readouts": [{"readout": str, "deviation": float, "weight": float, "mechanism": str}],
                "direction": str,  # "increase" / "decrease"
                "reasoning": str,
            }]
        """
        # Step 1: 计算所有Readout偏离度
        deviations = {}
        for readout, value in current_readouts.items():
            dev = cls.compute_readout_deviation(readout, value)
            if dev > 0.1:  # 只关注有意义的偏离
                deviations[readout] = dev

        if not deviations:
            return []

        # Step 2: 反向遍历因果图，累积每个Setpoint的调控收益
        setpoint_benefits = defaultdict(lambda: {
            "total_benefit": 0.0,
            "affected_readouts": [],
            "directions": [],
        })

        for (sp, ro), edge in cls.CAUSAL_EDGES.items():
            if ro not in deviations:
                continue
            dev = deviations[ro]
            weight = edge["weight"]
            priority = cls.SETPOINT_PRIORITY.get(sp, 10)

            # Benefit = 偏离度 × 因果权重 × 优先级倒数（优先级高的收益更大）
            benefit = dev * weight * (1.0 / priority)

            setpoint_benefits[sp]["total_benefit"] += benefit
            setpoint_benefits[sp]["affected_readouts"].append({
                "readout": ro,
                "deviation": round(dev, 3),
                "weight": weight,
                "direction": edge["direction"],
                "mechanism": edge["mechanism"],
            })
            setpoint_benefits[sp]["directions"].append(edge["direction"])

        # Step 3: 构建结果并排序
        results = []
        for sp, data in setpoint_benefits.items():
            if data["total_benefit"] < 0.01:
                continue

            # 推断调整方向
            direction_votes = data["directions"]
            increase_count = sum(1 for d in direction_votes if d == "positive")
            decrease_count = sum(1 for d in direction_votes if d == "negative")
            suggested_dir = "increase" if increase_count >= decrease_count else "decrease"

            # 生成推理文本
            top_readouts = sorted(data["affected_readouts"], key=lambda x: x["deviation"], reverse=True)[:3]
            reasoning_parts = []
            for ar in top_readouts:
                reasoning_parts.append(f"{ar['readout']}偏离{ar['deviation']:.0%}(w={ar['weight']:.1f}): {ar['mechanism']}")

            results.append({
                "setpoint": sp,
                "benefit_score": round(data["total_benefit"], 4),
                "priority": cls.SETPOINT_PRIORITY.get(sp, 10),
                "affected_readouts": data["affected_readouts"],
                "direction": suggested_dir,
                "reasoning": "; ".join(reasoning_parts),
                "readout_count": len(data["affected_readouts"]),
            })

        # 按 benefit_score 降序
        results.sort(key=lambda x: x["benefit_score"], reverse=True)
        return results


# =============================================================================
# 2. CUSUM Early Warning System
# =============================================================================

class CUSUMDetector:
    """
    累积和控制图 (CUSUM) 早期预警算法

    核心思想:
    - 标准阈值只在指标已经越界时才报警（滞后）
    - CUSUM通过累积微小偏差来检测趋势性恶化，实现"预警"
    - 当累积偏差超过决策区间h时触发预警，此时指标可能还未突破阈值

    算法:
    - S_n = max(0, S_{n-1} + (x_n - μ_0 - k))  (上侧CUSUM，检测升高)
    - T_n = max(0, T_{n-1} + (μ_0 - k - x_n))  (下侧CUSUM，检测降低)
    - 当 S_n > h 或 T_n > h 时，触发预警

    参数:
    - μ_0: 目标值（target midpoint）
    - k: 允许偏差（allowable slack），通常为 0.5σ
    - h: 决策区间（decision interval），通常为 4σ 或 5σ
    """

    @staticmethod
    def detect(time_series: List[float], target: float,
               k: float = None, h: float = None,
               higher_is_worse: bool = True) -> Dict[str, Any]:
        """
        对单个指标的时间序列执行CUSUM检测

        Args:
            time_series: 按时间排序的测量值 [t0, t1, t2, ...]
            target: 目标值 μ_0
            k: 允许偏差（None则自动计算为0.5σ）
            h: 决策区间（None则自动计算为4σ）
            higher_is_worse: True=检测升高趋势，False=检测降低趋势

        Returns:
            {
                "alarm": bool,          # 是否触发预警
                "alarm_side": str,      # "upper" / "lower" / None
                "cusum_upper": [...],   # 上侧CUSUM序列
                "cusum_lower": [...],   # 下侧CUSUM序列
                "trend": str,           # "worsening" / "improving" / "stable"
                "severity": float,      # 0-1 预警严重程度
                "message": str,         # 预警描述
            }
        """
        n = len(time_series)
        if n < 2:
            return {"alarm": False, "alarm_side": None, "cusum_upper": [], "cusum_lower": [],
                    "trend": "stable", "severity": 0, "message": "数据不足"}

        # 自动计算参数
        if k is None or h is None:
            sigma = max(0.001, _std(time_series))
            if k is None:
                k = 0.5 * sigma
            if h is None:
                h = 4.0 * sigma

        # 计算双侧CUSUM
        S = [0.0]  # 上侧（检测升高）
        T = [0.0]  # 下侧（检测降低）

        for i in range(n):
            x = time_series[i]
            s_new = max(0, S[-1] + (x - target - k))
            t_new = max(0, T[-1] + (target - k - x))
            S.append(s_new)
            T.append(t_new)

        S = S[1:]  # 去掉初始0
        T = T[1:]

        # 判断预警
        max_s = max(S) if S else 0
        max_t = max(T) if T else 0
        alarm_upper = max_s > h
        alarm_lower = max_t > h

        # 趋势判断
        if higher_is_worse:
            alarm = alarm_upper
            alarm_side = "upper" if alarm_upper else ("lower" if alarm_lower else None)
            worsening = alarm_upper
        else:
            alarm = alarm_lower
            alarm_side = "lower" if alarm_lower else ("upper" if alarm_upper else None)
            worsening = alarm_lower

        if worsening:
            trend = "worsening"
        elif (not higher_is_worse and alarm_upper) or (higher_is_worse and alarm_lower):
            trend = "improving"
        else:
            trend = "stable"

        # 严重程度
        relevant_cusum = max_s if higher_is_worse else max_t
        severity = min(1.0, relevant_cusum / h) if h > 0 else 0

        # 生成消息
        if alarm:
            message = f"CUSUM预警: 检测到持续{'升高' if alarm_side == 'upper' else '降低'}趋势 (严重度{severity:.0%})"
        elif severity > 0.5:
            message = f"CUSUM注意: 趋势偏离中 (严重度{severity:.0%}，尚未触发预警)"
        else:
            message = "CUSUM正常: 指标趋势平稳"

        return {
            "alarm": alarm,
            "alarm_side": alarm_side,
            "cusum_upper": [round(s, 3) for s in S],
            "cusum_lower": [round(t, 3) for t in T],
            "trend": trend,
            "severity": round(severity, 3),
            "message": message,
            "threshold_h": round(h, 3),
        }

    @staticmethod
    def batch_detect(time_series_dict: Dict[str, List[float]],
                     targets: Dict[str, float],
                     higher_worse_flags: Dict[str, bool] = None) -> Dict[str, Dict]:
        """
        批量检测多个指标

        Args:
            time_series_dict: {"Lactate": [2.8, 3.5, 4.1, ...], ...}
            targets: {"Lactate": 2.0, ...}
            higher_worse_flags: {"Lactate": True, ...}

        Returns:
            {"Lactate": {CUSUM result}, ...}
        """
        if higher_worse_flags is None:
            higher_worse_flags = {}

        results = {}
        for indicator, series in time_series_dict.items():
            target = targets.get(indicator)
            if target is None:
                continue
            hiw = higher_worse_flags.get(indicator, True)
            results[indicator] = CUSUMDetector.detect(series, target, higher_is_worse=hiw)
        return results


# =============================================================================
# 3. CPRS — Composite Perfusion Risk Score
# =============================================================================

class CompositePerfusionRiskScore:
    """
    复合灌注风险评分 (CPRS)

    核心思想:
    - 将所有指标归一化到统一尺度
    - 基于指标类型和干预优先级分配权重
    - 融合为单一0-100分数值：0=完美灌注，100=极高风险
    - 提供分项评分（灌注参数/心功能/代谢/氧合）支持临床解读

    加权策略:
    - Entropy权重: 基于指标变异性自适应调整（变异大的信息量多，权重高）
    - Priority权重: 基于干预优先级（高优先级指标权重高）
    - 最终权重 = normalize(entropy_w × priority_w)
    """

    # 指标分组（用于分项评分）
    DOMAINS = {
        "perfusion": {
            "indicators": ["Flow", "AoDP", "Temperature"],
            "label": "灌注参数",
        },
        "cardiac": {
            "indicators": ["EF", "CI", "dPdt_max", "SW"],
            "label": "心功能",
        },
        "metabolic": {
            "indicators": ["Lactate", "pH", "K_A"],
            "label": "代谢状态",
        },
        "oxygenation": {
            "indicators": ["PaO2", "Hemoglobin", "SvO2", "MVO2"],
            "label": "氧合状态",
        },
    }

    # 指标配置: target range + 权重优先级
    INDICATOR_WEIGHTS = {
        # Setpoints（权重略低，因为它们是可控的）
        "Flow":        {"target": (4.2, 4.8), "priority_w": 0.9, "higher_worse": False},
        "Temperature": {"target": (34, 37),   "priority_w": 0.7, "higher_worse": False},
        "AoDP":        {"target": (35, 45),   "priority_w": 0.8, "higher_worse": False},
        "PaO2":        {"target": (100, 200), "priority_w": 0.6, "higher_worse": False},
        "Hemoglobin":  {"target": (40, 50),   "priority_w": 0.7, "higher_worse": False},
        # Readouts（权重更高，反映灌注质量）
        "Lactate":     {"target": (0, 4.0),   "priority_w": 1.0, "higher_worse": True},
        "pH":          {"target": (7.25, 7.35), "priority_w": 0.8, "higher_worse": False},
        "K_A":         {"target": (3.5, 5.0), "priority_w": 0.9, "higher_worse": True},
        "EF":          {"target": (18, 60),   "priority_w": 0.85, "higher_worse": False},
        "CI":          {"target": (2.2, 4.0), "priority_w": 0.8, "higher_worse": False},
        "SvO2":        {"target": (65, 80),   "priority_w": 0.6, "higher_worse": False},
        "MVO2":        {"target": (8.8, 20),  "priority_w": 0.7, "higher_worse": False},
        "dPdt_max":    {"target": (1200, 1800), "priority_w": 0.7, "higher_worse": False},
    }

    @classmethod
    def _normalize_indicator(cls, indicator: str, value: float) -> float:
        """
        将指标值归一化为风险分 ∈ [0, 1]
        0 = 在目标范围内 (理想)
        1 = 严重偏离 (极端)
        """
        cfg = cls.INDICATOR_WEIGHTS.get(indicator)
        if not cfg:
            return 0.0
        lo, hi = cfg["target"]
        span = hi - lo if hi != lo else 1.0

        if lo <= value <= hi:
            return 0.0  # 在目标范围内 → 无风险
        elif value > hi:
            excess = (value - hi) / span
            return min(1.0, excess)
        else:
            deficit = (lo - value) / span
            return min(1.0, deficit)

    @classmethod
    def compute(cls, measurements: Dict[str, float]) -> Dict[str, Any]:
        """
        计算复合灌注风险评分 (CPRS)

        Args:
            measurements: {"Lactate": 4.5, "pH": 7.28, "Flow": 4.3, ...}

        Returns:
            {
                "total_score": float,       # 0-100 总风险分
                "risk_level": str,          # "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
                "domain_scores": {...},     # 分项评分
                "indicator_risks": [...],   # 每个指标的风险贡献
                "top_risks": [...],         # 风险最高的指标排名
            }
        """
        indicator_risks = []
        total_weighted_risk = 0.0
        total_weight = 0.0

        for indicator, value in measurements.items():
            cfg = cls.INDICATOR_WEIGHTS.get(indicator)
            if cfg is None:
                continue
            risk = cls._normalize_indicator(indicator, value)
            weight = cfg["priority_w"]

            indicator_risks.append({
                "indicator": indicator,
                "value": value,
                "risk": round(risk, 3),
                "weight": weight,
                "contribution": round(risk * weight, 4),
            })

            total_weighted_risk += risk * weight
            total_weight += weight

        # 总分 (0-100)
        total_score = (total_weighted_risk / total_weight * 100) if total_weight > 0 else 0
        total_score = min(100, round(total_score, 1))

        # 风险等级
        if total_score >= 60:
            risk_level = "CRITICAL"
        elif total_score >= 35:
            risk_level = "HIGH"
        elif total_score >= 15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # 分项评分
        domain_scores = {}
        for domain_key, domain_cfg in cls.DOMAINS.items():
            domain_risk = 0.0
            domain_weight = 0.0
            for ind in domain_cfg["indicators"]:
                if ind in measurements:
                    cfg = cls.INDICATOR_WEIGHTS.get(ind)
                    if cfg:
                        r = cls._normalize_indicator(ind, measurements[ind])
                        w = cfg["priority_w"]
                        domain_risk += r * w
                        domain_weight += w
            score = (domain_risk / domain_weight * 100) if domain_weight > 0 else 0
            domain_scores[domain_key] = {
                "label": domain_cfg["label"],
                "score": min(100, round(score, 1)),
            }

        # 风险排名
        indicator_risks.sort(key=lambda x: x["contribution"], reverse=True)
        top_risks = [ir for ir in indicator_risks if ir["risk"] > 0][:5]

        return {
            "total_score": total_score,
            "risk_level": risk_level,
            "domain_scores": domain_scores,
            "indicator_risks": indicator_risks,
            "top_risks": top_risks,
        }


# =============================================================================
# 4. Lactate Trajectory Prediction
# =============================================================================

class LactateTrajectoryPredictor:
    """
    乳酸清除轨迹预测

    核心思想:
    - 乳酸清除遵循近似指数衰减: L(t) = L_final + (L_0 - L_final) × e^{-k×t}
    - 通过最小二乘拟合估计清除速率 k
    - 预测达到目标乳酸值的时间
    - 清除率 >10%/h 被认为是良好预后指标

    临床应用:
    - OCS灌注: Lactate < 5 mmol/L 为可接受标准
    - 理想: Lactate 持续下降且清除率 > 10%/h
    - 预警: Lactate上升或清除率 < 5%/h
    """

    @staticmethod
    def predict(time_points: List[float], lactate_values: List[float],
                target: float = 2.0,
                acceptable: float = 5.0) -> Dict[str, Any]:
        """
        拟合乳酸清除曲线并预测

        Args:
            time_points: 时间点（分钟）[0, 60, 120, ...]
            lactate_values: 对应乳酸值 [3.9, 3.5, 3.0, ...]
            target: 理想乳酸目标值 (mmol/L)
            acceptable: 可接受乳酸上限 (mmol/L)

        Returns:
            {
                "clearance_rate": float,        # 清除率 (%/h)
                "clearance_quality": str,        # "good" / "marginal" / "poor" / "worsening"
                "predicted_target_time": float,  # 预测达标时间 (min)
                "predicted_values": [...],       # 预测轨迹
                "trend": str,                    # "clearing" / "plateau" / "rising"
                "k_decay": float,                # 衰减常数
                "half_life": float,              # 半衰期 (min)
                "message": str,
            }
        """
        n = len(time_points)
        if n < 2 or len(lactate_values) != n:
            return {"clearance_rate": 0, "clearance_quality": "unknown",
                    "predicted_target_time": None, "predicted_values": [],
                    "trend": "unknown", "k_decay": 0, "half_life": float('inf'),
                    "message": "数据不足"}

        L0 = lactate_values[0]
        L_last = lactate_values[-1]
        t_last = time_points[-1]
        t_first = time_points[0]
        dt_hours = (t_last - t_first) / 60.0

        # 简单清除率 (%/h)
        if L0 > 0 and dt_hours > 0:
            clearance_rate = ((L0 - L_last) / L0) * 100.0 / dt_hours
        else:
            clearance_rate = 0.0

        # 指数衰减拟合: L(t) = L_inf + (L0 - L_inf) * exp(-k*t)
        # 假设 L_inf = target (渐近线)
        L_inf = target
        k_decay = 0.0
        half_life = float('inf')

        if L0 > L_inf and L_last > L_inf:
            # ln((L(t) - L_inf) / (L0 - L_inf)) = -k * t
            # 使用最小二乘法拟合k
            sum_ty = 0.0
            sum_tt = 0.0
            count = 0
            for i in range(n):
                ratio = (lactate_values[i] - L_inf) / (L0 - L_inf)
                if ratio > 0:
                    y = -math.log(ratio)
                    t = (time_points[i] - t_first)
                    if t > 0:
                        sum_ty += t * y
                        sum_tt += t * t
                        count += 1

            if sum_tt > 0:
                k_decay = sum_ty / sum_tt  # min^{-1}
                if k_decay > 0:
                    half_life = math.log(2) / k_decay  # min

        # 趋势判断
        if n >= 3:
            # 看最近3个点的趋势
            recent = lactate_values[-3:] if n >= 3 else lactate_values
            diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff < -0.1:
                trend = "clearing"
            elif avg_diff > 0.1:
                trend = "rising"
            else:
                trend = "plateau"
        elif L_last < L0:
            trend = "clearing"
        elif L_last > L0:
            trend = "rising"
        else:
            trend = "plateau"

        # 清除质量评估
        if trend == "rising":
            clearance_quality = "worsening"
        elif clearance_rate > 10:
            clearance_quality = "good"
        elif clearance_rate > 5:
            clearance_quality = "marginal"
        else:
            clearance_quality = "poor"

        # 预测达标时间
        predicted_target_time = None
        if k_decay > 0 and L_last > target:
            # L(t) = L_inf + (L0 - L_inf) * exp(-k*t)
            # target = L_inf + (L0 - L_inf) * exp(-k*t_target)
            # 这里从当前点预测
            ratio = (target - L_inf) / (L_last - L_inf) if L_last > L_inf else 0
            if 0 < ratio < 1:
                predicted_target_time = -math.log(ratio) / k_decay  # min from now

        # 生成预测轨迹（未来240分钟）
        predicted_values = []
        if k_decay > 0:
            for t_future in range(0, 241, 15):
                pred = L_inf + (L_last - L_inf) * math.exp(-k_decay * t_future)
                predicted_values.append({
                    "time_min": t_last + t_future,
                    "predicted_lactate": round(pred, 2),
                })

        # 生成消息
        if clearance_quality == "good":
            message = f"乳酸清除良好 (清除率{clearance_rate:.1f}%/h, 半衰期{half_life:.0f}min)"
        elif clearance_quality == "marginal":
            message = f"乳酸清除边缘 (清除率{clearance_rate:.1f}%/h), 考虑优化灌注参数"
        elif clearance_quality == "worsening":
            message = f"乳酸持续上升! 清除率{clearance_rate:.1f}%/h, 需立即干预"
        else:
            message = f"乳酸清除缓慢 (清除率{clearance_rate:.1f}%/h), 建议增加Flow/Hb"

        if predicted_target_time is not None:
            message += f" · 预测{predicted_target_time:.0f}min后达到{target}mmol/L"

        return {
            "clearance_rate": round(clearance_rate, 2),
            "clearance_quality": clearance_quality,
            "predicted_target_time": round(predicted_target_time, 1) if predicted_target_time else None,
            "predicted_values": predicted_values,
            "trend": trend,
            "k_decay": round(k_decay, 6),
            "half_life": round(half_life, 1),
            "message": message,
            "current_lactate": L_last,
            "target": target,
        }


# =============================================================================
# Helper functions
# =============================================================================

def _std(values: List[float]) -> float:
    """计算标准差"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


# =============================================================================
# 便捷接口: 一次调用全部算法
# =============================================================================

def run_all_algorithms(current_data: Dict[str, float],
                       baseline_data: Dict[str, float] = None,
                       time_series: Dict[str, List[float]] = None,
                       time_points: List[float] = None) -> Dict[str, Any]:
    """
    一站式运行全部4个算法

    Args:
        current_data: 当前时间点的所有指标值
        baseline_data: 基线数据（可选）
        time_series: 时间序列 {"Lactate": [2.8, 3.5, ...], ...}（可选）
        time_points: 时间点列表 [0, 60, 120, ...]（可选）

    Returns:
        {
            "srco": Setpoint调控建议,
            "cprs": 复合风险评分,
            "cusum": CUSUM预警结果,
            "lactate_prediction": 乳酸轨迹预测,
        }
    """
    results = {}

    # 1. SRCO
    results["srco"] = SetpointReadoutCausalOptimizer.optimize(current_data)

    # 2. CPRS
    results["cprs"] = CompositePerfusionRiskScore.compute(current_data)

    # 3. CUSUM（需要时间序列）
    if time_series and time_points:
        targets = {
            "Lactate": 2.0, "pH": 7.30, "K_A": 4.25, "EF": 35,
            "Flow": 4.5, "Temperature": 37.0, "AoDP": 40,
        }
        hiw_flags = {
            "Lactate": True, "pH": False, "K_A": True, "EF": False,
            "Flow": False, "Temperature": False, "AoDP": False,
        }
        results["cusum"] = CUSUMDetector.batch_detect(time_series, targets, hiw_flags)
    else:
        results["cusum"] = {}

    # 4. 乳酸轨迹预测
    if time_series and time_points and "Lactate" in time_series:
        results["lactate_prediction"] = LactateTrajectoryPredictor.predict(
            time_points, time_series["Lactate"]
        )
    else:
        results["lactate_prediction"] = None

    return results
