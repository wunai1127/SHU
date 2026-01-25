#!/usr/bin/env python3
"""
测试灌注监测系统 - 使用真实移植数据
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.perfusion_monitor import PerfusionMonitor

# 列名映射: Excel列名 -> 系统指标名
COLUMN_MAPPING = {
    'LacA': 'Lactate',
    'MAP (mmHg)': 'MAP',
    'SvO2': 'SvO2',
    'K+A': 'K_A',
    'Na+A': 'Na_A',
    'GluA': 'GluA',
    'pO2V': 'pO2V',
    'CvO2 (mL/100mL)': 'CvO2',
    'AoF (L/min)': 'CF',  # 主动脉流量作为冠脉流量近似
    'pO2A': 'pO2A',
    'pHA': 'pH',
}

# 时间点映射 (注意Excel中有的时间点带trailing space)
TIME_MAPPING = {
    '30 min': 30,
    '30 min ': 30,
    '1 hour': 60,
    '1 hour ': 60,
    '2 hour': 120,
    '3 hour': 180,
    '4 hour': 240,
}


def load_sample_data(excel_path: str, sample_id: str) -> dict:
    """加载指定样本的时序数据"""
    df = pd.read_excel(excel_path)

    # 前向填充样本ID
    df['sample_id'] = df['Unnamed: 1'].ffill()

    # 筛选指定样本
    sample_df = df[df['sample_id'] == sample_id].copy()

    if sample_df.empty:
        raise ValueError(f"Sample {sample_id} not found")

    # 获取outcome
    outcome = sample_df['Success wean'].iloc[0]

    # 按时间点组织数据
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

    return {
        'sample_id': sample_id,
        'outcome': outcome,  # 0=失败, 1=成功
        'time_series': time_series
    }


def run_monitoring_session(sample_data: dict) -> str:
    """运行完整的监测会话"""
    monitor = PerfusionMonitor()

    results = []
    results.append(f"\n{'='*70}")
    results.append(f"样本: {sample_data['sample_id']}")
    results.append(f"实际结局: {'成功脱机' if sample_data['outcome'] == 1 else '脱机失败'}")
    results.append('='*70)

    time_series = sample_data['time_series']

    # t=30min 设置baseline
    if 30 in time_series:
        results.append("\n" + "="*70)
        results.append(">>> t=30min - 设置Baseline并立即评估")
        results.append("="*70)

        baseline_report = monitor.set_baseline(time_series[30], 30)
        results.append(baseline_report)

    # 后续时间窗监测
    for time_min in [60, 120, 180, 240]:
        if time_min in time_series:
            results.append("\n" + "-"*70)
            results.append(f">>> t={time_min}min - 时间窗监测")
            results.append("-"*70)

            # 处理时间窗数据
            monitor.process_time_window(time_series[time_min], time_min)
            # 生成报告
            report = monitor.generate_time_window_report(time_min)
            results.append(report)

    # 完整会话报告
    results.append("\n" + "="*70)
    results.append(">>> 完整会话总结")
    results.append("="*70)
    full_report = monitor.generate_full_session_report()
    results.append(full_report)

    return '\n'.join(results)


def main():
    excel_path = '移植特征筛选.xlsx'

    # 加载所有样本ID
    df = pd.read_excel(excel_path)
    df['sample_id'] = df['Unnamed: 1'].ffill()
    sample_ids = df['sample_id'].unique().tolist()

    print(f"发现 {len(sample_ids)} 个样本")
    print(f"样本列表: {sample_ids[:5]}...")
    print()

    # 测试第一个样本
    test_sample_id = sample_ids[0]
    print(f"测试样本: {test_sample_id}")
    print()

    try:
        sample_data = load_sample_data(excel_path, test_sample_id)

        print("=== 时序数据概览 ===")
        for time_min, measurements in sorted(sample_data['time_series'].items()):
            print(f"\nt={time_min}min:")
            for k, v in measurements.items():
                print(f"  {k}: {v}")

        print("\n\n" + "="*70)
        print("开始监测会话...")
        print("="*70)

        result = run_monitoring_session(sample_data)
        print(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
