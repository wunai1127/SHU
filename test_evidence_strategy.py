#!/usr/bin/env python3
"""
证据驱动策略测试 - 使用真实移植数据
展示 Baseline 时刻的精准策略推荐
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from src.evidence_strategy_engine import EvidenceStrategyEngine, BaselineStrategyReport

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
}


def load_baseline_data(excel_path: str) -> list:
    """加载所有样本的Baseline数据 (t=30min)"""
    df = pd.read_excel(excel_path)
    df['sample_id'] = df['Unnamed: 1'].ffill()

    samples = []
    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id].copy()
        if sample_df.empty:
            continue

        outcome = sample_df['Success wean'].iloc[0]

        # 只取30min的数据
        for _, row in sample_df.iterrows():
            time_str = row['Unnamed: 2']
            if time_str in ['30 min', '30 min ']:
                measurements = {}
                for excel_col, sys_col in COLUMN_MAPPING.items():
                    if excel_col in row and pd.notna(row[excel_col]):
                        measurements[sys_col] = float(row[excel_col])

                samples.append({
                    'sample_id': sample_id,
                    'outcome': outcome,
                    'measurements': measurements
                })
                break

    return samples


def run_evidence_strategy_test():
    """运行证据驱动策略测试"""
    print("=" * 100)
    print("EVIDENCE-DRIVEN BASELINE STRATEGY TEST")
    print("=" * 100)

    # 加载数据
    samples = load_baseline_data('移植特征筛选.xlsx')
    print(f"\n加载 {len(samples)} 个样本的 Baseline (t=30min) 数据")

    success = [s for s in samples if s['outcome'] == 1]
    failed = [s for s in samples if s['outcome'] == 0]
    print(f"成功脱机: {len(success)}, 失败脱机: {len(failed)}")

    # 创建策略引擎
    engine = EvidenceStrategyEngine()

    # 尝试连接Neo4j（如果可用）
    try:
        from src.neo4j_connector import Neo4jKnowledgeGraph
        neo4j = Neo4jKnowledgeGraph()
        engine.set_neo4j(neo4j)
        print("\n✓ Neo4j已连接，将使用知识图谱证据")
        has_neo4j = True
    except Exception as e:
        print(f"\n⚠ Neo4j未连接: {e}")
        print("  将使用预定义策略库")
        has_neo4j = False
        neo4j = None

    print("\n" + "=" * 100)

    # 选择几个典型样本进行详细分析
    # 1个失败样本 + 1个成功样本
    test_samples = []
    if failed:
        test_samples.append(('FAILED', failed[0]))
    if success:
        test_samples.append(('SUCCESS', success[0]))

    for outcome_label, sample in test_samples:
        print(f"\n{'#' * 100}")
        print(f"# SAMPLE: {sample['sample_id']} [{outcome_label}]")
        print(f"{'#' * 100}")

        # 生成策略报告
        report = engine.analyze_baseline(
            sample['measurements'],
            sample_id=sample['sample_id'],
            timestamp_min=30
        )

        # 输出详细报告
        print(engine.format_report(report))

        # 如果有Neo4j，展示证据详情
        if has_neo4j and report.recommendations:
            print("\n" + "=" * 60)
            print("KNOWLEDGE GRAPH EVIDENCE DETAILS")
            print("=" * 60)
            for rec in report.recommendations[:2]:
                if rec.supporting_evidence:
                    print(f"\n### {rec.indicator} Evidence Chain:")
                    for i, ev in enumerate(rec.supporting_evidence, 1):
                        print(f"  [{i}] {ev.statement}")
                        print(f"      Type: {ev.evidence_type.value}")
                        print(f"      Source: {ev.source_origin}")
                        print(f"      Relation: {ev.source_entity} --{ev.relation}--> {ev.target_entity}")

    # 统计所有样本的异常分布
    print("\n" + "=" * 100)
    print("ABNORMALITY DISTRIBUTION ACROSS ALL SAMPLES")
    print("=" * 100)

    abnormality_counts = {'success': {}, 'failed': {}}

    for sample in samples:
        report = engine.analyze_baseline(sample['measurements'], sample['sample_id'])
        group = 'success' if sample['outcome'] == 1 else 'failed'

        for abn in report.detected_abnormalities:
            key = f"{abn['indicator']}_{abn['direction']}"
            abnormality_counts[group][key] = abnormality_counts[group].get(key, 0) + 1

    print(f"\n{'Abnormality':<20} {'Success (n={})'.format(len(success)):<20} {'Failed (n={})'.format(len(failed)):<20}")
    print("-" * 60)

    all_keys = set(abnormality_counts['success'].keys()) | set(abnormality_counts['failed'].keys())
    for key in sorted(all_keys):
        s_count = abnormality_counts['success'].get(key, 0)
        f_count = abnormality_counts['failed'].get(key, 0)
        s_pct = s_count / len(success) * 100 if success else 0
        f_pct = f_count / len(failed) * 100 if failed else 0
        print(f"{key:<20} {s_count} ({s_pct:.0f}%){'':<10} {f_count} ({f_pct:.0f}%)")

    # 策略推荐统计
    print("\n" + "=" * 100)
    print("STRATEGY RECOMMENDATIONS SUMMARY")
    print("=" * 100)

    intervention_counts = {'success': {}, 'failed': {}}
    for sample in samples:
        report = engine.analyze_baseline(sample['measurements'], sample['sample_id'])
        group = 'success' if sample['outcome'] == 1 else 'failed'

        for rec in report.recommendations:
            interv = rec.intervention
            intervention_counts[group][interv] = intervention_counts[group].get(interv, 0) + 1

    print(f"\n{'Intervention':<40} {'Success':<15} {'Failed':<15}")
    print("-" * 70)

    all_interv = set(intervention_counts['success'].keys()) | set(intervention_counts['failed'].keys())
    for interv in sorted(all_interv):
        s = intervention_counts['success'].get(interv, 0)
        f = intervention_counts['failed'].get(interv, 0)
        print(f"{interv:<40} {s:<15} {f:<15}")

    # 关闭Neo4j
    if has_neo4j:
        neo4j.close()

    print("\n" + "=" * 100)
    print("TEST COMPLETED")
    print("=" * 100)


if __name__ == '__main__':
    run_evidence_strategy_test()
