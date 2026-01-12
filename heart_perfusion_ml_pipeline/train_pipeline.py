"""
Heart Perfusion ML Training Pipeline
完整的数据预处理到模型训练流程
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing.evhp_data_processor import EVHPDataProcessor, EVHPFeatureConfig
from models.time_series_model import LSTMAttentionModel, OutcomeClassifier, ModelConfig


def run_training_pipeline(
    data_file: str,
    output_dir: str,
    task: str = 'both',  # 'prediction', 'classification', 'both'
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    运行完整训练流程

    Args:
        data_file: EVHP数据Excel文件路径
        output_dir: 输出目录
        task: 任务类型
        test_size: 测试集比例
        random_state: 随机种子
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Heart Perfusion ML Training Pipeline")
    print("="*70)
    print(f"Data file: {data_file}")
    print(f"Output dir: {output_dir}")
    print(f"Task: {task}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    # ========== Step 1: Data Preprocessing ==========
    print("\n[Step 1] Data Preprocessing")
    print("-"*50)

    processor = EVHPDataProcessor(
        impute_method='knn',
        scale_method='standard'
    )

    results = processor.process_pipeline(
        file_path=data_file,
        output_dir=str(output_path / 'processed_data')
    )

    # ========== Step 2: Time Series Prediction ==========
    if task in ['prediction', 'both']:
        print("\n[Step 2] Time Series Prediction Model")
        print("-"*50)

        X_seq = results['sequence_data']['X']
        y_seq = results['sequence_data']['y']

        if len(X_seq) > 0:
            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq, y_seq,
                test_size=test_size,
                random_state=random_state
            )

            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Feature dim: {X_train.shape[-1]}")

            # 配置模型
            config = ModelConfig(
                input_dim=X_train.shape[-1],
                hidden_dim=64,
                num_layers=2,
                seq_length=X_train.shape[1],
                output_dim=1,
                dropout=0.2,
                attention_heads=4,
                learning_rate=1e-3,
                batch_size=16,
                epochs=50,
            )

            # 训练
            model = LSTMAttentionModel(config)

            try:
                history = model.train(
                    X_train, y_train,
                    X_test, y_test,
                    use_pytorch=True
                )

                # 评估
                predictions = model.predict(X_test)
                mae = np.mean(np.abs(predictions.flatten() - y_test))
                rmse = np.sqrt(np.mean((predictions.flatten() - y_test) ** 2))

                print(f"\nTest MAE: {mae:.4f}")
                print(f"Test RMSE: {rmse:.4f}")

                # 保存模型
                model.save(str(output_path / 'models' / 'lstm_attention'))

                # 保存训练历史
                with open(output_path / 'models' / 'lstm_training_history.json', 'w') as f:
                    json.dump(history, f)

            except Exception as e:
                print(f"PyTorch training failed: {e}")
                print("Skipping time series model training.")

        else:
            print("Insufficient sequence data for training.")

    # ========== Step 3: Outcome Classification ==========
    if task in ['classification', 'both']:
        print("\n[Step 3] Outcome Classification Model")
        print("-"*50)

        cls_data = results.get('classification_data')

        if cls_data and cls_data['X'] is not None:
            X_cls = cls_data['X']
            y_cls = cls_data['y']

            # 数据分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_cls, y_cls,
                test_size=test_size,
                random_state=random_state,
                stratify=y_cls if len(np.unique(y_cls)) > 1 else None
            )

            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Label distribution (train): {np.unique(y_train, return_counts=True)}")

            # 训练多个模型
            models_to_try = ['random_forest', 'logistic']

            best_model = None
            best_auc = 0

            for model_type in models_to_try:
                print(f"\nTraining {model_type}...")

                classifier = OutcomeClassifier(model_type=model_type)
                metrics = classifier.train(
                    X_train, y_train,
                    X_test, y_test
                )

                print(f"  Train AUC: {metrics.get('train_auc', 0):.4f}")
                print(f"  Test AUC: {metrics.get('val_auc', 0):.4f}")
                print(f"  Test Accuracy: {metrics.get('val_accuracy', 0):.4f}")

                if metrics.get('val_auc', 0) > best_auc:
                    best_auc = metrics.get('val_auc', 0)
                    best_model = classifier

            # 显示最佳模型的特征重要性
            if best_model:
                print(f"\nBest Model: {best_model.model_type} (AUC: {best_auc:.4f})")
                print("\nTop 10 Important Features:")
                for feat, imp in best_model.get_top_features(10):
                    print(f"  {feat}: {imp:.4f}")

        else:
            print("No classification data available (missing outcome labels).")

    # ========== Step 4: Save Results ==========
    print("\n[Step 4] Saving Results")
    print("-"*50)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_file': data_file,
        'statistics': results['statistics'],
        'feature_columns': results['feature_columns'],
    }

    with open(output_path / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")
    print("\n" + "="*70)
    print("Training Pipeline Complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Heart Perfusion ML Training Pipeline')
    parser.add_argument('--data', type=str, required=True, help='EVHP data Excel file')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--task', type=str, default='both',
                        choices=['prediction', 'classification', 'both'],
                        help='Training task')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    run_training_pipeline(
        data_file=args.data,
        output_dir=args.output,
        task=args.task,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == '__main__':
    main()
