# 心脏移植阈值管理系统

Heart Transplant Threshold Management System

## 概述

本系统提供心脏移植/机械灌注场景下的**特征阈值 + baseline + 策略**管理框架，支持：

- 阈值评估与分级（高/中/低信度）
- t=0 Baseline 对比分析
- 临床决策策略映射
- Pending 指标监测

## 目录结构

```
SHU/
├── config/                          # 配置文件
│   ├── thresholds.yaml              # 已确认阈值配置（含source/confidence）
│   ├── pending_indicators.yaml      # 待定指标配置（仅监测）
│   ├── baseline.yaml                # t=0 baseline配置
│   └── strategies.yaml              # 策略映射配置
├── knowledge_base/                  # 知识库文档
│   ├── confirmed_thresholds.md      # 已确认阈值汇总
│   ├── pending_indicators.md        # 待定指标汇总
│   └── sources.md                   # 数据来源说明
├── src/                             # Python模块
│   ├── __init__.py
│   ├── threshold_manager.py         # 阈值管理器
│   ├── baseline_evaluator.py        # Baseline评估器
│   └── strategy_mapper.py           # 策略映射器
└── README.md
```

## 快速使用

### 1. 评估单个指标

```python
from src.threshold_manager import ThresholdManager

manager = ThresholdManager()
result = manager.evaluate("EF", 48)

print(f"结果: {result.result.value}")  # gray_zone
print(f"置信度: {result.confidence.value}")  # high
print(f"建议: {result.action}")  # 综合评估
```

### 2. Baseline 对比

```python
from src.baseline_evaluator import BaselineEvaluator

evaluator = BaselineEvaluator()
comparison = evaluator.compare("Lactate", 3.5)

print(f"当前: {comparison.current_value}")
print(f"基线: {comparison.baseline_value}")
print(f"变化: {comparison.delta} ({comparison.delta_percent}%)")
print(f"趋势: {comparison.trend.value}")
```

### 3. 生成临床决策报告

```python
from src.strategy_mapper import StrategyMapper

mapper = StrategyMapper()

measurements = {
    "EF": 48,
    "CI": 1.9,
    "MAP": 58,
    "SvO2": 52,
    "Lactate": 4.5,
    "Emax": 120,  # pending指标
}

report = mapper.generate_decision_report(measurements)
print(report)
```

## 指标分类

### 已确认阈值（可用作通用 baseline）

| 指标 | 置信度 | 目标/接受 | 红线/拒绝 |
|------|--------|-----------|-----------|
| EF | 高 | >50% | <40% |
| CI | 中 | 2.2-2.8 | <2.0 |
| MAP | 中 | 65-80 | <60 |
| AOP | 高 | 40-100 (OCS) | <70 警告 |
| SvO₂ | 中 | ≥65% | <60%, <50%危急 |
| Lactate | 高 | <2理想, <5接受 | >5不合格 |
| HR | 高 | 60-100 | 110-120警戒 |
| PVR | 高 | <2.5 | >3.0 |
| CF | 高 | 400-900 (OCS) | - |

### 待定指标（Pending，仅监测）

- **心肌力学**: Emax, Max dP/dt, Tau, End systolic pressure, Potential energy, Contraction time, Developed pressure
- **代谢/电解质**: GluA, GluV, Na⁺A, K⁺A, LacA, CvO₂
- **血气**: pO₂V (低信度占位)

## 策略映射

```
红线触发 → 立即干预，升级支持
警戒/灰区 → 微调方案，加强监测
正常范围 → 维持当前方案
Pending → 仅监测记录，不做决策
```

### 升级策略示例

```
SvO₂ < 50% 或 CI下降 且 复苏无效
    → 正性肌力药物/肺血管扩张 ± MCS
```

## 数据来源

| 来源 | 状态 | 覆盖指标 |
|------|------|----------|
| Prot_000 OCS Protocol | ✅ 已获取 | AOP, CF |
| 临床共识/标准 | ✅ 已采用 | EF, CI, MAP, SvO₂, Lactate, HR, PVR |
| ISHLT 2022 Guidelines | ❌ 待获取 | 全面覆盖 |
| PROCEED II Trial | ❌ 待获取 | 器官灌注参数 |
| RCVSIM Model | ⚠️ 仅参考 | 心肌力学参数（非临床红线）|

## 待办事项

- [ ] 获取 ISHLT 2022 指南完整版
- [ ] 获取 PROCEED II、FDA SSED 等文献
- [ ] 为 pending 指标补齐阈值（文献验证或数据驱动）
- [ ] 建立定期复核机制

## 依赖

- Python 3.8+
- PyYAML

```bash
pip install pyyaml
```
