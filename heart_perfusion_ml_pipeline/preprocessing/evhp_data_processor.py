"""
EVHP Data Processor - 离体心脏灌注数据预处理模块
处理猪心EVHP实验的时序生理数据
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EVHPFeatureConfig:
    """EVHP特征配置"""

    # 动脉血气指标
    ARTERIAL_BLOOD_GAS = {
        'pH_art': 7,
        'pO2_art': 8,
        'Na_art': 9,
        'K_art': 10,
        'Ca_art': 11,
        'Cl_art': 12,
        'Glu_art': 13,
        'Lac_art': 14,      # 乳酸 - 关键指标
        'tHb_art': 15,
        'HCO3_art': 16,
        'O2SAT_art': 17,
    }

    # 静脉血气指标
    VENOUS_BLOOD_GAS = {
        'pH_ven': 18,
        'pO2_ven': 19,
        'Na_ven': 20,
        'K_ven': 21,
        'Ca_ven': 22,
        'Cl_ven': 23,
        'Glu_ven': 24,
        'Lac_ven': 25,
        'tHb_ven': 26,
        'HCO3_ven': 27,
        'O2SAT_ven': 28,
    }

    # 血流动力学参数
    HEMODYNAMIC = {
        'MAP': 29,          # 平均动脉压 (mmHg)
        'AoF': 30,          # 主动脉流量 (L/min)
        'CVR': 31,          # 冠脉血管阻力
    }

    # 代谢参数
    METABOLIC = {
        'MVO2': 32,         # 心肌氧耗量
        'Lac_Extrac': 33,   # 乳酸提取率 - 关键指标
        'CaO2': 34,         # 动脉氧含量
        'CvO2': 35,         # 静脉氧含量
    }

    # 心脏功能参数
    CARDIAC_FUNCTION = {
        'ESPVR': 36,        # 收缩末期压力-容积关系
        'EDPVR': 37,        # 舒张末期压力-容积关系
        'PRSW': 38,         # 搏出功
        'dPdtmax_EdV': 39,  # 压力变化率
        'Emax': 40,         # 最大弹性
        'V0': 41,
        'EDP': 42,          # 舒张末压
        'ESP': 43,          # 收缩末压
        'Dev_Pressure': 44, # 发展压
        'Sys_Eject_Period': 45,
        'Dias_Fill_Period': 46,
        'Mean_Sys_Press': 47,
        'Mean_Dias_Press': 48,
        'Contract_Time': 49,
        'Relax_Time': 50,
        'Max_dPdt': 51,
        'Min_dPdt': 52,
        'Contract_Index': 53,
        'Tau': 54,          # 时间常数
        'Max_Vol': 55,
        'Min_Vol': 56,
        'EDV': 57,          # 舒张末期容积
        'ESV': 58,          # 收缩末期容积
        'SV': 59,           # 每搏量
        'CO': 60,           # 心输出量
        'EF': 61,           # 射血分数 - 关键指标
        'Stroke_Work': 62,
        'Ea': 63,           # 动脉弹性
        'Max_Vent_Power': 64,
        'PAMP': 65,
        'PE': 66,           # 位能
        'PVA': 67,          # 压力-容积面积
        'Efficiency': 68,   # 效率
        'PE_MEC': 69,
        'HR': 70,           # 心率
        'CI': 71,           # 心脏指数
    }

    # 结局变量
    OUTCOME = {
        'Success_Wean': 72,  # 成功脱机 - 主要结局
    }

    # 基础信息
    BASIC_INFO = {
        'Pig_Weight': 4,
        'Heart_Weight': 5,
        'Post_EVHP_Weight': 6,
    }


class EVHPDataProcessor:
    """
    EVHP数据处理器
    处理离体心脏灌注实验的时序数据
    """

    def __init__(
        self,
        config: EVHPFeatureConfig = None,
        impute_method: str = 'knn',
        scale_method: str = 'standard',
    ):
        self.config = config or EVHPFeatureConfig()
        self.impute_method = impute_method
        self.scale_method = scale_method

        self.scaler = None
        self.imputer = None
        self.feature_names = []

        # 时间点映射
        self.time_mapping = {
            '0 min': 0, '0 min ': 0,
            '30 min': 0.5, '30 min ': 0.5,
            '1 hour': 1, '1 hour ': 1,
            '2 hour': 2, '2 hour ': 2,
            '3 hour': 3, '3 hour ': 3,
            '4 hour': 4, '4 hour ': 4,
            '5 hour': 5, '5 hour ': 5,
            'PostTX': 6,
        }

    def load_excel(self, file_path: str) -> pd.DataFrame:
        """
        加载Excel数据文件

        Args:
            file_path: Excel文件路径

        Returns:
            原始DataFrame
        """
        df = pd.read_excel(file_path, sheet_name=0, skiprows=1)
        print(f"Loaded data shape: {df.shape}")
        return df

    def parse_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析原始数据，提取结构化字段

        Args:
            df: 原始DataFrame

        Returns:
            解析后的DataFrame
        """
        # 构建列名映射
        all_features = {}
        all_features.update(self.config.BASIC_INFO)
        all_features.update(self.config.ARTERIAL_BLOOD_GAS)
        all_features.update(self.config.VENOUS_BLOOD_GAS)
        all_features.update(self.config.HEMODYNAMIC)
        all_features.update(self.config.METABOLIC)
        all_features.update(self.config.CARDIAC_FUNCTION)
        all_features.update(self.config.OUTCOME)

        # 提取数据
        data = {}
        data['exp_id'] = df.iloc[:, 2]  # 实验ID
        data['time_raw'] = df.iloc[:, 3]  # 时间点

        for feat_name, col_idx in all_features.items():
            if col_idx < df.shape[1]:
                data[feat_name] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')

        result_df = pd.DataFrame(data)

        # 转换时间为数值
        result_df['time_hours'] = result_df['time_raw'].map(self.time_mapping)

        # 前向填充实验ID
        result_df['exp_id'] = result_df['exp_id'].ffill()

        # 移除全空行
        result_df = result_df.dropna(how='all', subset=list(all_features.keys()))

        print(f"Parsed data shape: {result_df.shape}")
        print(f"Experiments: {result_df['exp_id'].nunique()}")

        return result_df

    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算派生特征

        Args:
            df: 基础数据DataFrame

        Returns:
            添加派生特征后的DataFrame
        """
        df = df.copy()

        # 1. 乳酸变化率 (Lactate dynamics)
        df['Lac_delta'] = df.groupby('exp_id')['Lac_art'].diff()
        df['Lac_pct_change'] = df.groupby('exp_id')['Lac_art'].pct_change()

        # 2. 乳酸清除率 (每小时)
        df['Lac_clearance'] = -df['Lac_delta'] / df.groupby('exp_id')['time_hours'].diff()

        # 3. 氧摄取率
        if 'O2SAT_art' in df.columns and 'O2SAT_ven' in df.columns:
            df['O2_extraction'] = df['O2SAT_art'] - df['O2SAT_ven']

        # 4. 动静脉乳酸差
        if 'Lac_art' in df.columns and 'Lac_ven' in df.columns:
            df['Lac_av_diff'] = df['Lac_art'] - df['Lac_ven']

        # 5. 心脏做功效率派生
        if 'CO' in df.columns and 'MAP' in df.columns:
            df['Cardiac_Power'] = df['CO'] * df['MAP'] / 451  # 心脏功率指数

        # 6. pH稳定性（标准差）
        df['pH_stability'] = df.groupby('exp_id')['pH_art'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )

        # 7. 趋势特征（最近3个时间点的趋势）
        for col in ['Lac_art', 'MAP', 'EF']:
            if col in df.columns:
                df[f'{col}_trend'] = df.groupby('exp_id')[col].transform(
                    lambda x: x.rolling(window=3, min_periods=1).apply(
                        lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0
                    )
                )

        print(f"Added derived features, new shape: {df.shape}")
        return df

    def impute_missing(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        填充缺失值

        Args:
            df: DataFrame
            feature_cols: 需要填充的特征列

        Returns:
            填充后的DataFrame
        """
        df = df.copy()

        if self.impute_method == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
            df[feature_cols] = self.imputer.fit_transform(df[feature_cols])
        elif self.impute_method == 'median':
            for col in feature_cols:
                df[col] = df[col].fillna(df[col].median())
        elif self.impute_method == 'forward':
            df[feature_cols] = df.groupby('exp_id')[feature_cols].ffill()
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

        return df

    def scale_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        特征缩放

        Args:
            df: DataFrame
            feature_cols: 需要缩放的特征列

        Returns:
            缩放后的DataFrame
        """
        df = df.copy()

        if self.scale_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            return df

        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        seq_length: int = 4,
        target_col: str = 'Lac_art',
        prediction_horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建时序序列数据（用于LSTM/Transformer）

        Args:
            df: 数据DataFrame
            feature_cols: 特征列
            seq_length: 序列长度
            target_col: 目标列
            prediction_horizon: 预测步长

        Returns:
            (X_sequences, y_targets, experiment_ids)
        """
        X_sequences = []
        y_targets = []
        exp_ids = []

        for exp_id, group in df.groupby('exp_id'):
            group = group.sort_values('time_hours')
            features = group[feature_cols].values
            targets = group[target_col].values

            # 创建滑动窗口
            for i in range(len(group) - seq_length - prediction_horizon + 1):
                X_sequences.append(features[i:i+seq_length])
                y_targets.append(targets[i+seq_length+prediction_horizon-1])
                exp_ids.append(exp_id)

        return (
            np.array(X_sequences),
            np.array(y_targets),
            np.array(exp_ids)
        )

    def create_classification_dataset(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        outcome_col: str = 'Success_Wean',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        创建分类数据集（预测成功脱机）

        Args:
            df: 数据DataFrame
            feature_cols: 特征列
            outcome_col: 结局列

        Returns:
            (X_features, y_labels, experiment_ids)
        """
        # 为每个实验聚合特征
        agg_funcs = {col: ['mean', 'std', 'min', 'max', 'last'] for col in feature_cols}

        # 按实验聚合
        aggregated = df.groupby('exp_id').agg(agg_funcs)
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns]

        # 获取结局标签
        outcomes = df.groupby('exp_id')[outcome_col].last()

        # 合并
        dataset = aggregated.join(outcomes)
        dataset = dataset.dropna(subset=[outcome_col])

        X = dataset.drop(columns=[outcome_col]).values
        y = dataset[outcome_col].values
        exp_ids = dataset.index.values

        return X, y, exp_ids

    def get_critical_features(self) -> List[str]:
        """获取关键临床特征列表"""
        return [
            # 血气关键指标
            'Lac_art', 'pH_art', 'pO2_art', 'O2SAT_art',
            'Lac_ven', 'pH_ven',
            # 血流动力学
            'MAP', 'AoF', 'CVR',
            # 代谢
            'MVO2', 'Lac_Extrac', 'CaO2', 'CvO2',
            # 心功能
            'EF', 'CO', 'CI', 'HR', 'Stroke_Work',
            'Max_dPdt', 'Min_dPdt', 'Tau',
            # 派生特征
            'Lac_delta', 'Lac_clearance', 'O2_extraction',
            'pH_stability', 'Lac_art_trend',
        ]

    def process_pipeline(
        self,
        file_path: str,
        output_dir: str = None,
    ) -> Dict[str, Any]:
        """
        完整处理流程

        Args:
            file_path: 输入Excel文件路径
            output_dir: 输出目录

        Returns:
            处理结果字典
        """
        # 1. 加载数据
        print("Step 1: Loading data...")
        raw_df = self.load_excel(file_path)

        # 2. 解析结构
        print("Step 2: Parsing data structure...")
        parsed_df = self.parse_raw_data(raw_df)

        # 3. 计算派生特征
        print("Step 3: Computing derived features...")
        enriched_df = self.compute_derived_features(parsed_df)

        # 4. 获取特征列
        feature_cols = [c for c in self.get_critical_features() if c in enriched_df.columns]
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:10]}...")

        # 5. 缺失值填充
        print("Step 4: Imputing missing values...")
        imputed_df = self.impute_missing(enriched_df, feature_cols)

        # 6. 特征缩放
        print("Step 5: Scaling features...")
        scaled_df = self.scale_features(imputed_df, feature_cols)

        # 7. 创建数据集
        print("Step 6: Creating datasets...")

        # 时序预测数据集
        X_seq, y_seq, exp_seq = self.create_sequences(
            scaled_df, feature_cols,
            seq_length=4, target_col='Lac_art', prediction_horizon=1
        )

        # 分类数据集（如果有结局变量）
        X_cls, y_cls, exp_cls = None, None, None
        if 'Success_Wean' in scaled_df.columns:
            X_cls, y_cls, exp_cls = self.create_classification_dataset(
                scaled_df, feature_cols, 'Success_Wean'
            )

        # 8. 保存结果
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 保存处理后的数据
            scaled_df.to_csv(output_path / 'processed_data.csv', index=False)

            # 保存序列数据
            np.save(output_path / 'X_sequences.npy', X_seq)
            np.save(output_path / 'y_sequences.npy', y_seq)

            if X_cls is not None:
                np.save(output_path / 'X_classification.npy', X_cls)
                np.save(output_path / 'y_classification.npy', y_cls)

            print(f"Results saved to {output_dir}")

        results = {
            'processed_df': scaled_df,
            'feature_columns': feature_cols,
            'sequence_data': {
                'X': X_seq,
                'y': y_seq,
                'exp_ids': exp_seq,
                'shape': X_seq.shape,
            },
            'classification_data': {
                'X': X_cls,
                'y': y_cls,
                'exp_ids': exp_cls,
            } if X_cls is not None else None,
            'statistics': {
                'n_experiments': scaled_df['exp_id'].nunique(),
                'n_timepoints': len(scaled_df),
                'n_features': len(feature_cols),
                'missing_rate': (1 - scaled_df[feature_cols].notna().mean()).to_dict(),
            }
        }

        print("\n" + "="*60)
        print("Processing Complete!")
        print("="*60)
        print(f"Experiments: {results['statistics']['n_experiments']}")
        print(f"Total timepoints: {results['statistics']['n_timepoints']}")
        print(f"Features: {results['statistics']['n_features']}")
        print(f"Sequence shape: {X_seq.shape}")
        if X_cls is not None:
            print(f"Classification shape: {X_cls.shape}, Labels: {np.unique(y_cls, return_counts=True)}")

        return results


if __name__ == "__main__":
    # 测试处理流程
    processor = EVHPDataProcessor(
        impute_method='knn',
        scale_method='standard'
    )

    # 处理数据
    # results = processor.process_pipeline(
    #     file_path="数据整理 - EVHP原始数据V1.xlsx",
    #     output_dir="outputs/evhp_processed"
    # )
