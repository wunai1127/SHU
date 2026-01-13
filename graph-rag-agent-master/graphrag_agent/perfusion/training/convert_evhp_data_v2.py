#!/usr/bin/env python3
"""
EVHP Data Converter V2 - 改进版数据转换

针对小数据集的优化：
1. 更合理的标签生成
2. 数据增强
3. 更多特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def convert_evhp_to_training_format(
    xlsx_path: str,
    output_path: str = './perfusion_training_data.csv',
    augment: bool = True,
    augment_factor: int = 5
):
    """
    转换 EVHP Excel 数据为训练格式（改进版）

    Args:
        xlsx_path: Excel 文件路径
        output_path: 输出 CSV 路径
        augment: 是否进行数据增强
        augment_factor: 数据增强倍数
    """
    print(f"Reading {xlsx_path}...")
    df = pd.read_excel(xlsx_path, header=None)

    # 列映射
    column_mapping = {
        2: 'case_id',
        3: 'time_point',
        7: 'pH',
        8: 'PO2',
        9: 'Na_plus',
        10: 'K_plus',
        14: 'lactate',
        29: 'MAP_mmHg',
        30: 'AoF_L_min',
        32: 'MVO2',
        60: 'cardiac_output',
        61: 'ejection_fraction',
        70: 'heart_rate',
    }

    # 提取数据
    data_rows = []
    current_case = None

    for idx in range(2, len(df)):
        row = df.iloc[idx]

        if pd.notna(row[2]):
            current_case = str(row[2]).strip()

        if current_case is None:
            continue

        row_data = {'case_id': current_case}

        for col_idx, col_name in column_mapping.items():
            if col_idx < len(row) and col_name != 'case_id':
                val = row[col_idx]
                if pd.notna(val):
                    try:
                        row_data[col_name] = float(val) if col_name != 'time_point' else val
                    except:
                        row_data[col_name] = val

        # 时间点映射
        time_str = str(row_data.get('time_point', ''))
        time_map = {
            '0 min': 0, '30 min': 1, '1 hour': 2, '2 hour': 3,
            '3 hour': 4, '4 hour': 5, '5 hour': 6, 'PostTX': 7
        }
        row_data['timestamp'] = -1
        for key, val in time_map.items():
            if key in time_str:
                row_data['timestamp'] = val
                break

        if row_data['timestamp'] >= 0:
            data_rows.append(row_data)

    result_df = pd.DataFrame(data_rows)

    # 填充缺失值（用中位数）
    numeric_cols = ['pH', 'PO2', 'Na_plus', 'K_plus', 'lactate',
                   'MAP_mmHg', 'AoF_L_min', 'MVO2', 'cardiac_output',
                   'ejection_fraction', 'heart_rate']

    for col in numeric_cols:
        if col in result_df.columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            result_df[col] = result_df[col].fillna(result_df[col].median())

    # 改进的标签生成（基于多个指标的综合评分）
    def calculate_quality_score(group):
        """基于最后时间点的多指标综合评分"""
        last_row = group.iloc[-1]
        first_row = group.iloc[0]

        score = 50  # 基础分（中等）

        # pH 评分 (权重: 25分)
        ph = last_row.get('pH')
        if pd.notna(ph):
            if 7.35 <= ph <= 7.45:
                score += 25
            elif 7.30 <= ph < 7.35 or 7.45 < ph <= 7.50:
                score += 15
            elif 7.25 <= ph < 7.30:
                score += 5
            else:
                score -= 10

        # 乳酸评分 (权重: 25分) - 越低越好
        lac = last_row.get('lactate')
        if pd.notna(lac):
            if lac < 2:
                score += 25
            elif lac < 3:
                score += 15
            elif lac < 4:
                score += 5
            elif lac < 6:
                score -= 5
            else:
                score -= 15

        # 乳酸趋势 (权重: 10分)
        lac_first = first_row.get('lactate')
        if pd.notna(lac) and pd.notna(lac_first):
            if lac < lac_first:  # 下降趋势好
                score += 10
            elif lac > lac_first * 1.5:  # 上升超过50%差
                score -= 10

        # 射血分数 (权重: 15分)
        ef = last_row.get('ejection_fraction')
        if pd.notna(ef):
            if ef > 50:
                score += 15
            elif ef > 40:
                score += 10
            elif ef > 30:
                score += 5
            else:
                score -= 10

        # 心输出量 (权重: 10分)
        co = last_row.get('cardiac_output')
        if pd.notna(co):
            if co > 4:
                score += 10
            elif co > 3:
                score += 5
            elif co < 2:
                score -= 5

        # MAP 评分 (权重: 10分)
        map_val = last_row.get('MAP_mmHg')
        if pd.notna(map_val):
            if 60 <= map_val <= 80:
                score += 10
            elif 50 <= map_val < 60 or 80 < map_val <= 90:
                score += 5
            else:
                score -= 5

        return np.clip(score, 0, 100)

    # 为每个病例计算标签
    case_labels = {}
    for case_id, group in result_df.groupby('case_id'):
        quality = calculate_quality_score(group)

        # 基于质量分数确定风险等级（更均衡的分布）
        if quality >= 75:
            risk_level = 'low'
        elif quality >= 55:
            risk_level = 'medium'
        elif quality >= 35:
            risk_level = 'high'
        else:
            risk_level = 'critical'

        case_labels[case_id] = {
            'quality_score': quality,
            'risk_level': risk_level,
            'usable': 1 if quality >= 50 else 0
        }

    # 添加标签
    result_df['quality_score'] = result_df['case_id'].map(lambda x: case_labels[x]['quality_score'])
    result_df['risk_level'] = result_df['case_id'].map(lambda x: case_labels[x]['risk_level'])
    result_df['usable'] = result_df['case_id'].map(lambda x: case_labels[x]['usable'])

    print(f"\n原始数据统计:")
    print(f"  病例数: {result_df['case_id'].nunique()}")
    print(f"  总行数: {len(result_df)}")
    print(f"  风险分布: {result_df.groupby('case_id')['risk_level'].first().value_counts().to_dict()}")
    print(f"  质量分数范围: {result_df['quality_score'].min():.1f} - {result_df['quality_score'].max():.1f}")

    # 数据增强
    if augment:
        print(f"\n进行数据增强 (x{augment_factor})...")
        augmented_rows = []

        for case_id, group in result_df.groupby('case_id'):
            # 原始数据
            augmented_rows.append(group)

            # 添加噪声生成新样本
            for aug_idx in range(augment_factor - 1):
                augmented = group.copy()
                augmented['case_id'] = f"{case_id}_aug{aug_idx}"

                # 对数值列添加小噪声
                for col in numeric_cols:
                    if col in augmented.columns:
                        std = augmented[col].std()
                        if pd.notna(std) and std > 0:
                            noise = np.random.normal(0, std * 0.05, len(augmented))
                            augmented[col] = augmented[col] + noise

                # 质量分数也添加小扰动
                augmented['quality_score'] = augmented['quality_score'] + np.random.normal(0, 3)
                augmented['quality_score'] = augmented['quality_score'].clip(0, 100)

                augmented_rows.append(augmented)

        result_df = pd.concat(augmented_rows, ignore_index=True)
        print(f"  增强后病例数: {result_df['case_id'].nunique()}")
        print(f"  增强后总行数: {len(result_df)}")

    # 保存
    result_df.to_csv(output_path, index=False)
    print(f"\n保存到: {output_path}")

    return result_df


if __name__ == '__main__':
    import sys

    xlsx_path = sys.argv[1] if len(sys.argv) > 1 else './数据整理 - EVHP原始数据V1.xlsx'

    # 使用数据增强
    convert_evhp_to_training_format(
        xlsx_path,
        augment=True,
        augment_factor=10  # 10倍增强: 32 -> 320 病例
    )
