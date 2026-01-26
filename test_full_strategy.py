#!/usr/bin/env python3
"""
完整策略测试 - Baseline(30min) → 60min → 120min 策略演变
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.perfusion_monitor import PerfusionMonitor
from src.baseline_thresholds import BaselineThresholds

# 列名映射
COLUMN_MAPPING = {
    'LacA': 'Lactate',
    'MAP (mmHg)': 'MAP',
    'SvO2': 'SvO2',
    'K+A': 'K_A',
    'Na+A': 'Na_A',
    'GluA': 'GluA',
    'pO2V': 'pO2V',
    'CvO2 (mL/100mL)': 'CvO2',
    'AoF (L/min)': 'CF',
    'pO2A': 'pO2A',
    'pHA': 'pH',
}

TIME_MAPPING = {
    '30 min': 30, '30 min ': 30,
    '1 hour': 60, '1 hour ': 60,
    '2 hour': 120,
    '3 hour': 180,
    '4 hour': 240,
}


def load_all_samples(excel_path: str) -> list:
    """加载所有样本数据"""
    df = pd.read_excel(excel_path)
    df['sample_id'] = df['Unnamed: 1'].ffill()

    samples = []
    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id].copy()
        if sample_df.empty:
            continue

        outcome = sample_df['Success wean'].iloc[0]

        time_series = {}
        for _, row in sample_df.iterrows():
            time_str = row['Unnamed: 2']
            if time_str in TIME_MAPPING:
                time_min = TIME_MAPPING[time_str]
                measurements = {}
                for excel_col, sys_col in COLUMN_MAPPING.items():
                    if excel_col in row and pd.notna(row[excel_col]):
                        measurements[sys_col] = float(row[excel_col])
                time_series[time_min] = measurements

        samples.append({
            'sample_id': sample_id,
            'outcome': outcome,
            'time_series': time_series
        })

    return samples


def format_strategy_summary(monitor, time_min: int, baseline_thresholds: BaselineThresholds, measurements: dict) -> str:
    """格式化策略摘要"""
    lines = []

    if not measurements:
        return f"  ⚠️ 无 t={time_min}min 数据"

    # 使用baseline_thresholds检查
    alerts = baseline_thresholds.get_alerts(measurements)

    critical = [a for a in alerts if a.alert_level.value == 'critical']
    red_line = [a for a in alerts if a.alert_level.value == 'red_line']
    warning = [a for a in alerts if a.alert_level.value == 'warning']

    # 状态汇总
    status_icons = []
    if critical:
        status_icons.append(f"🔴×{len(critical)}")
    if red_line:
        status_icons.append(f"🟠×{len(red_line)}")
    if warning:
        status_icons.append(f"🟡×{len(warning)}")
    if not status_icons:
        status_icons.append("🟢 正常")

    lines.append(f"  状态: {' '.join(status_icons)}")

    # 关键指标
    key_indicators = ['Lactate', 'MAP', 'SvO2', 'K_A', 'CvO2']
    key_values = []
    for ind in key_indicators:
        if ind in measurements:
            val = measurements[ind]
            result = baseline_thresholds.check_threshold(ind, val)
            icon = "🔴" if result.alert_level.value in ['critical', 'red_line'] else "🟡" if result.alert_level.value == 'warning' else "🟢"
            key_values.append(f"{ind}={val:.1f}{icon}")
    lines.append(f"  关键指标: {', '.join(key_values)}")

    # 干预建议
    interventions = []
    for alert in critical + red_line:
        intervention = baseline_thresholds.generate_evidence_enhanced_intervention(
            alert.indicator, alert.value, include_evidence=False
        )
        if intervention:
            interventions.append(f"{alert.indicator}: {intervention.intervention}")

    if interventions:
        lines.append(f"  干预建议:")
        for interv in interventions[:3]:
            lines.append(f"    → {interv}")

    return '\n'.join(lines)


def analyze_trend(baseline_val, current_val, indicator: str) -> str:
    """分析趋势"""
    if baseline_val is None or current_val is None:
        return "?"

    diff = current_val - baseline_val
    pct = (diff / baseline_val * 100) if baseline_val != 0 else 0

    # 乳酸降低是好的，MAP升高是好的
    good_decrease = ['Lactate']
    good_increase = ['MAP', 'SvO2', 'CF']

    if indicator in good_decrease:
        if diff < -0.5:
            return f"↓{abs(diff):.1f} 📈改善"
        elif diff > 0.5:
            return f"↑{diff:.1f} 📉恶化"
        else:
            return f"稳定"
    elif indicator in good_increase:
        if diff > 5:
            return f"↑{diff:.1f} 📈改善"
        elif diff < -5:
            return f"↓{abs(diff):.1f} 📉恶化"
        else:
            return f"稳定"
    else:
        return f"Δ{diff:+.1f}"


def run_full_test():
    """运行完整测试"""
    print("=" * 80)
    print("HTTG 灌注策略测试 - Baseline → 60min → 120min")
    print("=" * 80)

    # 加载数据
    samples = load_all_samples('移植特征筛选.xlsx')
    print(f"\n总样本数: {len(samples)}")

    success_samples = [s for s in samples if s['outcome'] == 1]
    failed_samples = [s for s in samples if s['outcome'] == 0]
    print(f"成功脱机: {len(success_samples)}, 失败脱机: {len(failed_samples)}")

    # 初始化
    baseline_thresholds = BaselineThresholds()

    # 统计
    stats = {
        'baseline_alerts': {'success': [], 'failed': []},
        '60min_alerts': {'success': [], 'failed': []},
        '120min_alerts': {'success': [], 'failed': []},
    }

    print("\n" + "=" * 80)
    print("详细样本分析")
    print("=" * 80)

    for sample in samples:
        monitor = PerfusionMonitor()
        outcome = 'success' if sample['outcome'] == 1 else 'failed'
        outcome_text = '✓成功' if sample['outcome'] == 1 else '✗失败'

        print(f"\n{'─' * 80}")
        print(f"样本: {sample['sample_id']} [{outcome_text}]")
        print('─' * 80)

        ts = sample['time_series']

        # ========== BASELINE (t=30min) ==========
        if 30 in ts:
            monitor.set_baseline(ts[30], 30)
            alerts_30 = baseline_thresholds.get_alerts(ts[30])
            stats['baseline_alerts'][outcome].append(len(alerts_30))

            print(f"\n▶ BASELINE (t=30min)")
            print(format_strategy_summary(monitor, 30, baseline_thresholds, ts[30]))

        # ========== 60min 策略 ==========
        if 60 in ts:
            monitor.process_time_window(ts[60], 60)
            alerts_60 = baseline_thresholds.get_alerts(ts[60])
            stats['60min_alerts'][outcome].append(len(alerts_60))

            print(f"\n▶ 60min 策略")
            print(format_strategy_summary(monitor, 60, baseline_thresholds, ts[60]))

            # 趋势分析
            if 30 in ts:
                trends = []
                for ind in ['Lactate', 'MAP', 'SvO2']:
                    if ind in ts[30] and ind in ts[60]:
                        trend = analyze_trend(ts[30][ind], ts[60][ind], ind)
                        trends.append(f"{ind}: {trend}")
                print(f"  趋势(vs baseline): {', '.join(trends)}")

        # ========== 120min 策略 ==========
        if 120 in ts:
            monitor.process_time_window(ts[120], 120)
            alerts_120 = baseline_thresholds.get_alerts(ts[120])
            stats['120min_alerts'][outcome].append(len(alerts_120))

            print(f"\n▶ 120min 策略")
            print(format_strategy_summary(monitor, 120, baseline_thresholds, ts[120]))

            # 趋势分析
            if 30 in ts:
                trends = []
                for ind in ['Lactate', 'MAP', 'SvO2']:
                    if ind in ts[30] and ind in ts[120]:
                        trend = analyze_trend(ts[30][ind], ts[120][ind], ind)
                        trends.append(f"{ind}: {trend}")
                print(f"  趋势(vs baseline): {', '.join(trends)}")

    # ========== 统计汇总 ==========
    print("\n" + "=" * 80)
    print("策略效果统计汇总")
    print("=" * 80)

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print("\n平均警报数量对比:")
    print(f"{'时间点':<15} {'成功组':<15} {'失败组':<15} {'差异':<15}")
    print("-" * 60)

    for key, label in [('baseline_alerts', 'Baseline'), ('60min_alerts', '60min'), ('120min_alerts', '120min')]:
        s_avg = avg(stats[key]['success'])
        f_avg = avg(stats[key]['failed'])
        diff = f_avg - s_avg
        print(f"{label:<15} {s_avg:<15.2f} {f_avg:<15.2f} {diff:+.2f}")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    # 计算关键指标差异
    baseline_s = avg(stats['baseline_alerts']['success'])
    baseline_f = avg(stats['baseline_alerts']['failed'])
    t120_s = avg(stats['120min_alerts']['success'])
    t120_f = avg(stats['120min_alerts']['failed'])

    print(f"""
HTTG框架分析结果:

1. Baseline阶段 (t=30min):
   - 成功组平均警报: {baseline_s:.1f}
   - 失败组平均警报: {baseline_f:.1f}
   - 差异: {baseline_f - baseline_s:+.1f} (失败组警报更{'多' if baseline_f > baseline_s else '少'})

2. 120min阶段:
   - 成功组平均警报: {t120_s:.1f}
   - 失败组平均警报: {t120_f:.1f}
   - 差异: {t120_f - t120_s:+.1f}

3. 策略演变:
   - 成功组警报变化: {baseline_s:.1f} → {t120_s:.1f} ({t120_s - baseline_s:+.1f})
   - 失败组警报变化: {baseline_f:.1f} → {t120_f:.1f} ({t120_f - baseline_f:+.1f})
""")


if __name__ == '__main__':
    run_full_test()
