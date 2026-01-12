#!/usr/bin/env python3
"""
EVHP Data Converter - 将 Excel 数据转换为 GNN 训练格式

将 EVHP 原始数据转换为训练所需的 CSV 格式
"""

import pandas as pd
import numpy as np
from pathlib import Path


def convert_evhp_to_training_format(
    xlsx_path: str,
    output_path: str = './perfusion_training_data.csv'
):
    """
    转换 EVHP Excel 数据为训练格式

    Args:
        xlsx_path: Excel 文件路径
        output_path: 输出 CSV 路径
    """
    print(f"Reading {xlsx_path}...")
    df = pd.read_excel(xlsx_path, header=None)

    # 定义列映射（根据 Excel 结构）
    # Column mapping based on Excel structure
    column_mapping = {
        2: 'case_id',       # Exp.
        3: 'time_point',    # Time
        4: 'pig_weight_kg', # pig weight
        5: 'heart_weight_g', # heart weight
        # Blood Gas - arterial
        7: 'pH',
        8: 'PO2',
        9: 'Na_plus',
        10: 'K_plus',
        11: 'Ca_plus',
        12: 'Cl_minus',
        13: 'glucose',
        14: 'lactate',
        15: 'tHb',
        16: 'HCO3',
        17: 'O2_sat',
        # Hemodynamic
        29: 'MAP_mmHg',
        30: 'AoF_L_min',     # 主动脉流量
        31: 'CVR',           # 冠脉阻力
        # Metabolic
        32: 'MVO2',
        33: 'lac_extraction',
        # Cardiac function
        36: 'ESPVR',
        37: 'EDPVR',
        38: 'PRSW',
        51: 'max_dPdt',
        52: 'min_dPdt',
        60: 'cardiac_output',
        61: 'ejection_fraction',
        62: 'stroke_work',
        70: 'heart_rate',
    }

    # 提取数据（跳过前2行表头）
    data_rows = []
    current_case = None

    for idx in range(2, len(df)):
        row = df.iloc[idx]

        # 更新当前病例ID
        if pd.notna(row[2]):
            current_case = str(row[2]).strip()

        if current_case is None:
            continue

        # 提取该行数据
        row_data = {'case_id': current_case}

        for col_idx, col_name in column_mapping.items():
            if col_idx < len(row):
                val = row[col_idx]
                if col_name != 'case_id':
                    row_data[col_name] = val

        # 转换时间点为数值
        time_str = str(row_data.get('time_point', ''))
        if '0 min' in time_str:
            row_data['timestamp'] = 0
        elif '30 min' in time_str:
            row_data['timestamp'] = 1
        elif '1 hour' in time_str:
            row_data['timestamp'] = 2
        elif '2 hour' in time_str:
            row_data['timestamp'] = 3
        elif '3 hour' in time_str:
            row_data['timestamp'] = 4
        elif '4 hour' in time_str:
            row_data['timestamp'] = 5
        elif '5 hour' in time_str:
            row_data['timestamp'] = 6
        elif 'PostTX' in time_str:
            row_data['timestamp'] = 7
        else:
            row_data['timestamp'] = -1

        if row_data['timestamp'] >= 0:
            data_rows.append(row_data)

    # 创建 DataFrame
    result_df = pd.DataFrame(data_rows)

    # 生成标签（基于规则）
    # Generate labels based on rules
    def calculate_quality_score(group):
        """根据最后时间点的指标计算质量分数"""
        last_row = group.iloc[-1]
        score = 70  # 基础分

        # pH 评分
        ph = last_row.get('pH')
        if pd.notna(ph):
            if ph >= 7.35 and ph <= 7.45:
                score += 10
            elif ph < 7.2 or ph > 7.5:
                score -= 20

        # 乳酸评分
        lac = last_row.get('lactate')
        if pd.notna(lac):
            if lac < 2:
                score += 10
            elif lac > 4:
                score -= 15
            elif lac > 6:
                score -= 30

        # 射血分数评分
        ef = last_row.get('ejection_fraction')
        if pd.notna(ef):
            if ef > 50:
                score += 10
            elif ef < 30:
                score -= 20

        return np.clip(score, 0, 100)

    def calculate_risk_level(quality_score):
        if quality_score >= 80:
            return 'low'
        elif quality_score >= 60:
            return 'medium'
        elif quality_score >= 40:
            return 'high'
        else:
            return 'critical'

    # 为每个病例计算标签
    case_labels = {}
    for case_id, group in result_df.groupby('case_id'):
        quality = calculate_quality_score(group)
        case_labels[case_id] = {
            'quality_score': quality,
            'risk_level': calculate_risk_level(quality),
            'usable': 1 if quality >= 60 else 0
        }

    # 添加标签到数据
    result_df['quality_score'] = result_df['case_id'].map(lambda x: case_labels[x]['quality_score'])
    result_df['risk_level'] = result_df['case_id'].map(lambda x: case_labels[x]['risk_level'])
    result_df['usable'] = result_df['case_id'].map(lambda x: case_labels[x]['usable'])

    # 重新排列列顺序
    cols = ['case_id', 'timestamp', 'time_point', 'pH', 'PO2', 'lactate',
            'K_plus', 'Na_plus', 'MAP_mmHg', 'AoF_L_min', 'heart_rate',
            'ejection_fraction', 'cardiac_output', 'quality_score', 'risk_level', 'usable']

    # 只保留存在的列
    cols = [c for c in cols if c in result_df.columns]
    result_df = result_df[cols]

    # 保存
    result_df.to_csv(output_path, index=False)
    print(f"\n转换完成！")
    print(f"  输出文件: {output_path}")
    print(f"  病例数量: {result_df['case_id'].nunique()}")
    print(f"  总行数: {len(result_df)}")
    print(f"\n数据预览:")
    print(result_df.head(10).to_string())

    return result_df


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        xlsx_path = sys.argv[1]
    else:
        xlsx_path = './数据整理 - EVHP原始数据V1.xlsx'

    convert_evhp_to_training_format(xlsx_path)
