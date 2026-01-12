"""
Agent 1 测试脚本
演示完整的输入理解流程
"""

import torch
import json
from pathlib import Path
from agent1_core import InputUnderstandingAgent


def load_example_data(example_path: str = 'examples/example_input.json'):
    """加载示例数据"""
    with open(example_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_agent1_basic():
    """基础测试: 验证Agent 1能正常处理输入"""
    print("=" * 80)
    print("Agent 1 基础测试")
    print("=" * 80)

    # 1. 创建Agent 1
    print("\n[1/4] 初始化 Agent 1...")
    agent = InputUnderstandingAgent()

    # 2. 加载示例数据
    print("\n[2/4] 加载示例输入数据...")
    raw_input = load_example_data()

    # 3. 处理输入
    print("\n[3/4] 处理输入数据...")
    standardized_output = agent.process(raw_input)

    # 4. 显示结果
    print("\n[4/4] 生成摘要...")
    print(agent.summary(standardized_output))

    print("\n" + "=" * 80)
    print("✅ Agent 1 基础测试通过!")
    print("=" * 80)

    return standardized_output


def test_text_encoder():
    """测试文本编码器"""
    print("\n" + "=" * 80)
    print("测试文本编码器")
    print("=" * 80)

    from agent1_core import ClinicalTextEncoder

    encoder = ClinicalTextEncoder()

    # 测试文本
    test_texts = [
        "Heart appears mildly hypertrophied with no visible scarring. Coronary arteries patent. Left ventricle shows good contractility.",
        "Severe hypertrophy noted. Multiple areas of scarring present. Poor contractility observed. Valve incompetence detected.",
        "Normal heart size. Excellent contractility. All valves competent. No visible damage."
    ]

    for i, text in enumerate(test_texts):
        print(f"\n📝 测试样本 {i+1}:")
        print(f"   文本: {text[:80]}...")

        embedding, features = encoder(text)

        print(f"\n   提取特征:")
        print(f"   • 肥厚程度: {features['hypertrophy_level']:.3f}")
        print(f"   • 收缩功能: {features['contractility_score']:.3f}")
        print(f"   • 瓣膜状态: {features['valve_status']} ({features['valve_status_score']:.3f})")
        print(f"   • 瘢痕程度: {features['scarring_level']:.3f}")
        print(f"   • 冠脉通畅: {features['coronary_patency']:.3f}")
        print(f"   • 可见损伤: {'是' if features['visible_damage'] else '否'}")

    print("\n✅ 文本编码器测试完成!")


def test_lstm_encoder():
    """测试LSTM编码器"""
    print("\n" + "=" * 80)
    print("测试LSTM时序编码器")
    print("=" * 80)

    from agent1_core import BloodGasLSTMEncoder

    encoder = BloodGasLSTMEncoder()

    # 测试序列 (模拟5个时间点的血气数据)
    test_sequences = [
        # 正常改善趋势
        torch.tensor([
            [2.8, 7.32, 280, 45, 4.2, 120],  # t=0
            [2.0, 7.36, 310, 43, 4.1, 115],  # t=60
            [1.4, 7.38, 330, 41, 4.0, 110],  # t=120
            [1.0, 7.40, 345, 40, 4.0, 105],  # t=180
            [0.7, 7.42, 360, 38, 3.9, 100],  # t=240
        ], dtype=torch.float32),

        # 异常恶化趋势
        torch.tensor([
            [2.5, 7.35, 300, 42, 4.0, 110],
            [2.8, 7.32, 290, 44, 4.3, 115],
            [3.2, 7.28, 275, 47, 4.6, 120],
            [3.6, 7.25, 260, 50, 5.0, 125],
            [4.0, 7.22, 245, 53, 5.4, 130],
        ], dtype=torch.float32)
    ]

    labels = ["改善趋势", "恶化趋势"]

    for i, sequence in enumerate(test_sequences):
        print(f"\n🔬 测试序列 {i+1}: {labels[i]}")

        # 编码
        embedding, attn_weights = encoder(sequence.unsqueeze(0))

        # 计算趋势特征
        trajectory = encoder.compute_trajectory_features(sequence)

        print(f"\n   嵌入维度: {embedding.shape}")
        print(f"\n   趋势特征:")
        print(f"   • 乳酸清除率: {trajectory['lactate_clearance_rate']:.4f} mmol/L/单位时间")
        print(f"   • 初始乳酸: {trajectory['lactate_initial']:.2f} mmol/L")
        print(f"   • 最终乳酸: {trajectory['lactate_final']:.2f} mmol/L")
        print(f"   • pH稳定性: {trajectory['ph_stability']:.3f}")
        print(f"   • pH均值: {trajectory['ph_mean']:.2f}")
        print(f"   • 氧合趋势: {trajectory['oxygenation_trend']}")
        print(f"   • pO2改善: {trajectory['po2_improvement']:.2f} mmHg/单位时间")
        print(f"   • K+稳定性: {trajectory['k_stability']:.2%}")

    print("\n✅ LSTM编码器测试完成!")


def test_strategy_extractor():
    """测试策略特征提取器"""
    print("\n" + "=" * 80)
    print("测试策略特征提取器")
    print("=" * 80)

    from agent1_core import StrategyFeatureExtractor

    extractor = StrategyFeatureExtractor()

    # 测试策略
    test_strategies = [
        {
            "method": "HTK solution",
            "temperature": 4,
            "pressure": 60,  # 偏低
            "flow_rate": 1.2,
            "duration": 240,
            "additives": ["adenosine", "insulin"],
            "delivery_mode": "antegrade"
        },
        {
            "method": "Del Nido",
            "temperature": 4,
            "pressure": 75,  # 正常
            "flow_rate": 1.4,
            "duration": 180,
            "additives": ["adenosine", "insulin", "magnesium"],
            "delivery_mode": "combined"
        }
    ]

    for i, strategy in enumerate(test_strategies):
        print(f"\n💉 策略 {i+1}:")
        print(f"   方法: {strategy['method']}")
        print(f"   压力: {strategy['pressure']} mmHg")
        print(f"   流速: {strategy['flow_rate']} L/min")

        vector, features = extractor.extract(strategy)

        print(f"\n   提取特征:")
        print(f"   • 方法评分: {features['method_score']:.2f}")
        print(f"   • 压力充分性: {features['pressure_adequacy']}")
        print(f"   • 压力归一化: {features['pressure_normalized']:.3f}")
        print(f"   • 温度最优: {'是' if features['temperature_optimal'] else '否'}")
        print(f"   • 流速充分性: {features['flow_adequacy']:.3f}")
        print(f"   • 添加剂数量: {features['num_additives']}")
        print(f"   • 递送模式: {features['delivery_mode']}")
        print(f"\n   特征向量维度: {vector.shape}")

    print("\n✅ 策略提取器测试完成!")


def test_patient_profiler():
    """测试患者风险画像器"""
    print("\n" + "=" * 80)
    print("测试患者风险画像器")
    print("=" * 80)

    from agent1_core import PatientRiskProfiler

    profiler = PatientRiskProfiler()

    # 测试患者
    test_patients = [
        {
            "demographics": {"age": 55, "gender": "male", "weight": 78, "height": 175},
            "comorbidities": ["diabetes", "hypertension", "CKD stage 3"],
            "lab_results": {
                "creatinine": 1.8,
                "BNP": 2400,
                "troponin": 0.15,
                "albumin": 3.5
            },
            "hemodynamics": {
                "LVEF": 15,
                "PVR": 3.2,
                "cardiac_output": 3.8,
                "PCWP": 28
            },
            "previous_interventions": ["LVAD", "ICD implantation"]
        },
        {
            "demographics": {"age": 45, "gender": "female", "weight": 65, "height": 165},
            "comorbidities": ["hypertension"],
            "lab_results": {
                "creatinine": 1.0,
                "BNP": 800,
                "troponin": 0.05,
                "albumin": 4.2
            },
            "hemodynamics": {
                "LVEF": 35,
                "PVR": 2.0,
                "cardiac_output": 4.8,
                "PCWP": 15
            },
            "previous_interventions": []
        }
    ]

    labels = ["高风险患者", "低风险患者"]

    for i, patient in enumerate(test_patients):
        print(f"\n👤 {labels[i]}:")
        print(f"   年龄: {patient['demographics']['age']}")
        print(f"   合并症: {', '.join(patient['comorbidities'])}")

        profile_vector, risk_factors = profiler.compute_profile(patient)

        print(f"\n   识别的风险因素:")
        for rf in risk_factors:
            print(f"   • {rf}")

        print(f"\n   风险因素总数: {len(risk_factors)}")
        print(f"   画像向量维度: {profile_vector.shape}")

    print("\n✅ 患者画像器测试完成!")


def test_full_pipeline():
    """完整流程测试"""
    print("\n" + "=" * 80)
    print("完整流程测试")
    print("=" * 80)

    # 运行完整测试
    standardized_output = test_agent1_basic()

    # 验证输出
    print("\n🔍 验证输出维度:")
    print(f"   • 文本嵌入: {standardized_output.cardiac_text_embedding.shape}")
    print(f"   • 血气嵌入: {standardized_output.blood_gas_embedding.shape}")
    print(f"   • 血气序列: {standardized_output.blood_gas_sequence.shape}")
    print(f"   • 策略参数: {standardized_output.strategy_params.shape}")
    print(f"   • 患者画像: {standardized_output.patient_profile.shape}")

    assert standardized_output.cardiac_text_embedding.shape == torch.Size([768]), "文本嵌入维度错误"
    assert standardized_output.blood_gas_embedding.shape == torch.Size([256]), "血气嵌入维度错误"
    assert standardized_output.strategy_params.shape == torch.Size([20]), "策略参数维度错误"
    assert standardized_output.patient_profile.shape == torch.Size([50]), "患者画像维度错误"

    print("\n✅ 所有维度验证通过!")

    # 保存输出
    output_path = "outputs/agent1_output.pt"
    Path("outputs").mkdir(exist_ok=True)
    torch.save({
        'cardiac_text_embedding': standardized_output.cardiac_text_embedding,
        'blood_gas_embedding': standardized_output.blood_gas_embedding,
        'blood_gas_sequence': standardized_output.blood_gas_sequence,
        'strategy_params': standardized_output.strategy_params,
        'patient_profile': standardized_output.patient_profile,
        'cardiac_features': standardized_output.cardiac_features,
        'metabolic_trajectory': standardized_output.metabolic_trajectory,
        'strategy_features': standardized_output.strategy_features,
        'risk_factors': standardized_output.risk_factors,
        'extracted_entities': standardized_output.extracted_entities
    }, output_path)

    print(f"\n💾 输出已保存至: {output_path}")

    return standardized_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Agent 1")
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'basic', 'text', 'lstm', 'strategy', 'patient'],
                       help='测试类型')

    args = parser.parse_args()

    if args.test == 'all':
        test_text_encoder()
        test_lstm_encoder()
        test_strategy_extractor()
        test_patient_profiler()
        test_full_pipeline()
    elif args.test == 'basic':
        test_agent1_basic()
    elif args.test == 'text':
        test_text_encoder()
    elif args.test == 'lstm':
        test_lstm_encoder()
    elif args.test == 'strategy':
        test_strategy_extractor()
    elif args.test == 'patient':
        test_patient_profiler()

    print("\n" + "=" * 80)
    print("🎉 所有测试完成!")
    print("=" * 80)
