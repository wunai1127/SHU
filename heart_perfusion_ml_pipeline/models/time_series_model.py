"""
Time Series Models - 时序预测模型
LSTM + Attention for EVHP physiological prediction
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """模型配置"""
    input_dim: int = 20          # 输入特征维度
    hidden_dim: int = 128        # LSTM隐藏层维度
    num_layers: int = 2          # LSTM层数
    output_dim: int = 1          # 输出维度
    dropout: float = 0.2         # Dropout率
    attention_heads: int = 4     # 注意力头数
    seq_length: int = 4          # 序列长度
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100


class LSTMAttentionModel:
    """
    LSTM + Attention 时序预测模型

    用于预测：
    1. 乳酸水平变化
    2. pH变化趋势
    3. 血流动力学指标
    """

    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.history = None

    def build_model_pytorch(self):
        """
        构建PyTorch模型

        Architecture:
        - Bidirectional LSTM layers
        - Multi-head Self-Attention
        - Fully connected output layers
        """
        try:
            import torch
            import torch.nn as nn

            class LSTMAttention(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config

                    # Bidirectional LSTM
                    self.lstm = nn.LSTM(
                        input_size=config.input_dim,
                        hidden_size=config.hidden_dim,
                        num_layers=config.num_layers,
                        batch_first=True,
                        dropout=config.dropout if config.num_layers > 1 else 0,
                        bidirectional=True
                    )

                    # Multi-head Attention
                    self.attention = nn.MultiheadAttention(
                        embed_dim=config.hidden_dim * 2,  # bidirectional
                        num_heads=config.attention_heads,
                        dropout=config.dropout,
                        batch_first=True
                    )

                    # Output layers
                    self.fc = nn.Sequential(
                        nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.hidden_dim, config.output_dim)
                    )

                    # Layer normalization
                    self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)

                def forward(self, x):
                    # LSTM encoding
                    lstm_out, _ = self.lstm(x)

                    # Self-attention
                    attn_out, attn_weights = self.attention(
                        lstm_out, lstm_out, lstm_out
                    )

                    # Residual connection + LayerNorm
                    out = self.layer_norm(lstm_out + attn_out)

                    # Use last timestep for prediction
                    out = out[:, -1, :]

                    # Output projection
                    output = self.fc(out)

                    return output, attn_weights

            self.model = LSTMAttention(self.config)
            return self.model

        except ImportError:
            print("PyTorch not available, using numpy implementation")
            return None

    def build_model_tensorflow(self):
        """
        构建TensorFlow/Keras模型 (备选)
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            # Input
            inputs = keras.Input(shape=(self.config.seq_length, self.config.input_dim))

            # Bidirectional LSTM
            x = layers.Bidirectional(
                layers.LSTM(self.config.hidden_dim, return_sequences=True, dropout=self.config.dropout)
            )(inputs)

            x = layers.Bidirectional(
                layers.LSTM(self.config.hidden_dim, return_sequences=True, dropout=self.config.dropout)
            )(x)

            # Attention
            attention = layers.MultiHeadAttention(
                num_heads=self.config.attention_heads,
                key_dim=self.config.hidden_dim
            )(x, x)

            x = layers.Add()([x, attention])
            x = layers.LayerNormalization()(x)

            # Global average pooling
            x = layers.GlobalAveragePooling1D()(x)

            # Output
            x = layers.Dense(self.config.hidden_dim, activation='relu')(x)
            x = layers.Dropout(self.config.dropout)(x)
            outputs = layers.Dense(self.config.output_dim)(x)

            self.model = keras.Model(inputs, outputs)
            self.model.compile(
                optimizer=keras.optimizers.Adam(self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            return self.model

        except ImportError:
            print("TensorFlow not available")
            return None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        use_pytorch: bool = True,
    ) -> Dict[str, List[float]]:
        """
        训练模型

        Args:
            X_train: 训练数据 (N, seq_len, features)
            y_train: 训练标签 (N,)
            X_val: 验证数据
            y_val: 验证标签
            use_pytorch: 是否使用PyTorch

        Returns:
            训练历史
        """
        if use_pytorch:
            return self._train_pytorch(X_train, y_train, X_val, y_val)
        else:
            return self._train_tensorflow(X_train, y_train, X_val, y_val)

    def _train_pytorch(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """PyTorch训练循环"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            # 设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")

            # 构建模型
            if self.model is None:
                self.build_model_pytorch()
            self.model = self.model.to(device)

            # 准备数据
            X_train_t = torch.FloatTensor(X_train).to(device)
            y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)

            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

            if X_val is not None:
                X_val_t = torch.FloatTensor(X_val).to(device)
                y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

            # 优化器和损失
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            # 训练历史
            history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

            # 训练循环
            for epoch in range(self.config.epochs):
                self.model.train()
                train_loss = 0
                train_mae = 0

                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs, _ = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    train_mae += torch.mean(torch.abs(outputs - batch_y)).item()

                train_loss /= len(train_loader)
                train_mae /= len(train_loader)

                history['loss'].append(train_loss)
                history['mae'].append(train_mae)

                # 验证
                if X_val is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs, _ = self.model(X_val_t)
                        val_loss = criterion(val_outputs, y_val_t).item()
                        val_mae = torch.mean(torch.abs(val_outputs - y_val_t)).item()

                    history['val_loss'].append(val_loss)
                    history['val_mae'].append(val_mae)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {train_loss:.4f} - MAE: {train_mae:.4f}", end='')
                    if X_val is not None:
                        print(f" - Val Loss: {val_loss:.4f} - Val MAE: {val_mae:.4f}")
                    else:
                        print()

            self.history = history
            return history

        except ImportError:
            print("PyTorch not available")
            return {}

    def _train_tensorflow(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """TensorFlow训练"""
        if self.model is None:
            self.build_model_tensorflow()

        validation_data = (X_val, y_val) if X_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            verbose=1
        )

        self.history = history.history
        return history.history

    def predict(self, X: np.ndarray, return_attention: bool = False) -> np.ndarray:
        """
        预测

        Args:
            X: 输入数据
            return_attention: 是否返回注意力权重

        Returns:
            预测结果
        """
        try:
            import torch
            self.model.eval()
            device = next(self.model.parameters()).device
            X_t = torch.FloatTensor(X).to(device)

            with torch.no_grad():
                outputs, attn_weights = self.model(X_t)

            if return_attention:
                return outputs.cpu().numpy(), attn_weights.cpu().numpy()
            return outputs.cpu().numpy()

        except:
            return self.model.predict(X)

    def save(self, path: str):
        """保存模型"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f)

        # 保存模型权重
        try:
            import torch
            torch.save(self.model.state_dict(), path / 'model.pt')
        except:
            self.model.save(path / 'model')

    def load(self, path: str):
        """加载模型"""
        path = Path(path)

        # 加载配置
        with open(path / 'config.json', 'r') as f:
            config_dict = json.load(f)
        self.config = ModelConfig(**config_dict)

        # 加载模型
        try:
            import torch
            self.build_model_pytorch()
            self.model.load_state_dict(torch.load(path / 'model.pt'))
        except:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(path / 'model')


class OutcomeClassifier:
    """
    结局分类器 - 预测脱机成功率

    使用聚合的时序特征进行二分类
    """

    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

    def build_model(self):
        """构建分类模型"""
        if self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    eval_metric='auc',
                    use_label_encoder=False,
                )
            except ImportError:
                self.model_type = 'random_forest'

        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        elif self.model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(max_iter=1000, random_state=42)

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
    ) -> Dict[str, float]:
        """训练分类器"""
        if self.model is None:
            self.build_model()

        self.model.fit(X_train, y_train)

        # 获取特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                feature_names or [f'f{i}' for i in range(X_train.shape[1])],
                self.model.feature_importances_
            ))

        # 评估
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

        train_pred = self.model.predict(X_train)
        train_prob = self.model.predict_proba(X_train)[:, 1]

        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_auc': roc_auc_score(y_train, train_prob) if len(np.unique(y_train)) > 1 else 0,
            'train_f1': f1_score(y_train, train_pred, zero_division=0),
        }

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_prob = self.model.predict_proba(X_val)[:, 1]

            metrics.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_auc': roc_auc_score(y_val, val_prob) if len(np.unique(y_val)) > 1 else 0,
                'val_f1': f1_score(y_val, val_pred, zero_division=0),
            })

        return metrics

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测"""
        pred = self.model.predict(X)
        prob = self.model.predict_proba(X)[:, 1]
        return pred, prob

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取最重要的特征"""
        if self.feature_importance is None:
            return []

        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: -x[1]
        )
        return sorted_features[:n]


if __name__ == "__main__":
    # 测试
    config = ModelConfig(
        input_dim=20,
        hidden_dim=64,
        num_layers=2,
        seq_length=4,
        epochs=10
    )

    model = LSTMAttentionModel(config)

    # 生成测试数据
    X_test = np.random.randn(100, 4, 20).astype(np.float32)
    y_test = np.random.randn(100).astype(np.float32)

    # 训练
    # history = model.train(X_test[:80], y_test[:80], X_test[80:], y_test[80:])
