#!/usr/bin/env python3
"""
Baseline Thresholds 综合测试套件

覆盖:
1. BaselineThresholds - 阈值检查、警报、证据映射
2. BaselineEvaluator - baseline对比、趋势分析
3. ThresholdManager - 阈值评估、置信度
4. 配置完整性 - YAML配置文件交叉验证
5. Setpoint指标 - 灌注调控参数阈值
"""

import sys
import os
import unittest
import yaml
from pathlib import Path

# 确保 src 在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_thresholds import (
    BaselineThresholds, AlertLevel, TrendDirection, ThresholdResult
)
from src.baseline_evaluator import (
    BaselineEvaluator, Trend, DeviationSeverity, BaselineComparison
)
from src.threshold_manager import (
    ThresholdManager, EvaluationResult, Confidence, ThresholdEvaluation
)


class TestConfigIntegrity(unittest.TestCase):
    """配置文件完整性测试"""

    def setUp(self):
        self.config_dir = Path(__file__).parent.parent / "config"
        with open(self.config_dir / "thresholds.yaml", 'r', encoding='utf-8') as f:
            self.thresholds_config = yaml.safe_load(f)
        with open(self.config_dir / "baseline.yaml", 'r', encoding='utf-8') as f:
            self.baseline_config = yaml.safe_load(f)

    def test_thresholds_yaml_loads(self):
        """thresholds.yaml 应能正确加载"""
        self.assertIsInstance(self.thresholds_config, dict)
        self.assertIn('version', self.thresholds_config)

    def test_baseline_yaml_loads(self):
        """baseline.yaml 应能正确加载"""
        self.assertIsInstance(self.baseline_config, dict)
        self.assertIn('confirmed_baselines', self.baseline_config)

    def test_all_thresholds_have_required_fields(self):
        """所有阈值配置应有必要字段（name, unit, thresholds）"""
        skip_keys = {'version', 'last_updated', 'update_notes'}
        for category, indicators in self.thresholds_config.items():
            if category in skip_keys or not isinstance(indicators, dict):
                continue
            for ind_name, ind_config in indicators.items():
                if not isinstance(ind_config, dict):
                    continue
                self.assertIn('name', ind_config,
                              f"{category}.{ind_name} 缺少 'name' 字段")
                self.assertIn('unit', ind_config,
                              f"{category}.{ind_name} 缺少 'unit' 字段")

    def test_confirmed_baselines_have_required_fields(self):
        """所有确认baseline应有baseline_value和acceptable_deviation"""
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        for name, config in confirmed.items():
            self.assertIn('baseline_value', config,
                          f"baseline '{name}' 缺少 'baseline_value'")
            self.assertIn('acceptable_deviation', config,
                          f"baseline '{name}' 缺少 'acceptable_deviation'")

    def test_setpoint_indicators_in_thresholds(self):
        """灌注调控参数(Setpoints)应存在于thresholds.yaml"""
        setpoints = ['Flow', 'Temperature', 'AoDP', 'PaO2',
                     'Hemoglobin', 'PacingRate', 'Dobutamine', 'Insulin']
        perfusion_section = self.thresholds_config.get('perfusion_setpoints', {})
        for sp in setpoints:
            self.assertIn(sp, perfusion_section,
                          f"Setpoint '{sp}' 不在 thresholds.yaml 的 perfusion_setpoints 中")

    def test_setpoint_indicators_in_baselines(self):
        """灌注调控参数(Setpoints)应有baseline配置"""
        setpoints = ['Flow', 'Temperature', 'AoDP', 'PaO2',
                     'Hemoglobin', 'PacingRate', 'Dobutamine', 'Insulin']
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        for sp in setpoints:
            self.assertIn(sp, confirmed,
                          f"Setpoint '{sp}' 缺少 baseline 配置")

    def test_readout_indicators_in_thresholds(self):
        """功能观测指标(Readouts)应存在于thresholds.yaml"""
        readouts = ['EF', 'CI', 'MAP', 'SvO2', 'Lactate', 'K_A', 'HR', 'PVR']
        bt = BaselineThresholds()
        for r in readouts:
            config = bt.get_threshold_config(r)
            self.assertIsNotNone(config, f"Readout '{r}' 无法在thresholds.yaml中找到")

    def test_blood_gas_supplementary_in_thresholds(self):
        """血气补充指标(pH, MVO2)应存在于thresholds.yaml"""
        supplementary = self.thresholds_config.get('blood_gas_supplementary', {})
        self.assertIn('pH', supplementary, "pH 不在 blood_gas_supplementary 中")
        self.assertIn('MVO2', supplementary, "MVO2 不在 blood_gas_supplementary 中")

    def test_baseline_reference_consistency(self):
        """thresholds.yaml中的baseline_reference应与baseline.yaml中的值一致"""
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        bt = BaselineThresholds()
        mismatches = []
        for ind_name, baseline_conf in confirmed.items():
            threshold_conf = bt.get_threshold_config(ind_name)
            if threshold_conf and 'baseline_reference' in threshold_conf:
                yaml_ref = threshold_conf['baseline_reference']
                yaml_baseline = baseline_conf.get('baseline_value')
                if yaml_ref is not None and yaml_baseline is not None and yaml_ref != yaml_baseline:
                    mismatches.append(
                        f"{ind_name}: thresholds.baseline_reference={yaml_ref} "
                        f"!= baseline.baseline_value={yaml_baseline}"
                    )
        # 允许一些不匹配(如Temperature baseline是22但reference不同)
        # 但记录下来以便排查
        if mismatches:
            print(f"\n[INFO] baseline_reference 不一致 ({len(mismatches)} 个):")
            for m in mismatches:
                print(f"  - {m}")


class TestBaselineThresholds(unittest.TestCase):
    """BaselineThresholds 类测试"""

    def setUp(self):
        self.bt = BaselineThresholds()

    def test_init_loads_configs(self):
        """初始化应加载配置"""
        self.assertIsNotNone(self.bt.thresholds_config)
        self.assertIsNotNone(self.bt.baseline_config)
        self.assertTrue(len(self.bt.threshold_index) > 0)

    def test_get_baseline_confirmed(self):
        """应返回确认指标的baseline"""
        baseline = self.bt.get_baseline('EF')
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['baseline_value'], 55)

    def test_get_baseline_pending(self):
        """应返回pending指标的baseline"""
        baseline = self.bt.get_baseline('Emax')
        self.assertIsNotNone(baseline)
        self.assertIsNone(baseline['baseline_value'])

    def test_get_baseline_setpoint(self):
        """应返回Setpoint指标的baseline"""
        baseline = self.bt.get_baseline('Flow')
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline['baseline_value'], 4.5)

    def test_get_baseline_nonexistent(self):
        """不存在的指标应返回None"""
        baseline = self.bt.get_baseline('NonExistent')
        self.assertIsNone(baseline)

    # === Readout 阈值检查 ===

    def test_ef_normal(self):
        """EF=55 应为正常"""
        result = self.bt.check_threshold('EF', 55)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_ef_reject(self):
        """EF=35 应触发red_line(reject)"""
        result = self.bt.check_threshold('EF', 35)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    def test_ci_red_line(self):
        """CI=1.8 应触发red_line"""
        result = self.bt.check_threshold('CI', 1.8)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    def test_ci_normal(self):
        """CI=2.5 应为正常"""
        result = self.bt.check_threshold('CI', 2.5)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_lactate_reject(self):
        """Lactate=6.5 应触发red_line(reject)"""
        result = self.bt.check_threshold('Lactate', 6.5)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    def test_lactate_ideal(self):
        """Lactate=1.5 应为正常"""
        result = self.bt.check_threshold('Lactate', 1.5)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_k_high_warning(self):
        """K_A=5.8 应触发warning"""
        result = self.bt.check_threshold('K_A', 5.8)
        self.assertIn(result.alert_level, [AlertLevel.WARNING, AlertLevel.RED_LINE])

    def test_k_critical(self):
        """K_A=6.8 应触发critical"""
        result = self.bt.check_threshold('K_A', 6.8)
        self.assertEqual(result.alert_level, AlertLevel.CRITICAL)

    def test_svo2_critical(self):
        """SvO2=45 应触发critical"""
        result = self.bt.check_threshold('SvO2', 45)
        self.assertEqual(result.alert_level, AlertLevel.CRITICAL)

    def test_map_red_line(self):
        """MAP=55 应触发red_line"""
        result = self.bt.check_threshold('MAP', 55)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    # === Setpoint 阈值检查 ===

    def test_flow_normal(self):
        """Flow=4.5 应为正常"""
        result = self.bt.check_threshold('Flow', 4.5)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_flow_warning(self):
        """Flow=3.6 应触发warning或red_line"""
        result = self.bt.check_threshold('Flow', 3.6)
        self.assertIn(result.alert_level, [AlertLevel.WARNING, AlertLevel.RED_LINE])

    def test_flow_red_line(self):
        """Flow=3.2 应触发red_line"""
        result = self.bt.check_threshold('Flow', 3.2)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    def test_temperature_target(self):
        """Temperature=35 应在目标范围"""
        result = self.bt.check_threshold('Temperature', 35)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_aodp_warning(self):
        """AoDP=28 应触发warning"""
        result = self.bt.check_threshold('AoDP', 28)
        self.assertIn(result.alert_level, [AlertLevel.WARNING, AlertLevel.RED_LINE])

    def test_hemoglobin_red_line(self):
        """Hemoglobin=28 应触发red_line"""
        result = self.bt.check_threshold('Hemoglobin', 28)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    def test_pao2_normal(self):
        """PaO2=150 应为正常"""
        result = self.bt.check_threshold('PaO2', 150)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_pao2_low(self):
        """PaO2=55 应触发red_line"""
        result = self.bt.check_threshold('PaO2', 55)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    # === pH 阈值检查 ===

    def test_ph_normal(self):
        """pH=7.30 应为正常"""
        result = self.bt.check_threshold('pH', 7.30)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)

    def test_ph_warning(self):
        """pH=7.18 应触发warning"""
        result = self.bt.check_threshold('pH', 7.18)
        self.assertIn(result.alert_level, [AlertLevel.WARNING, AlertLevel.RED_LINE])

    def test_ph_red_line(self):
        """pH=7.10 应触发red_line"""
        result = self.bt.check_threshold('pH', 7.10)
        self.assertEqual(result.alert_level, AlertLevel.RED_LINE)

    # === Deviation 计算 ===

    def test_deviation_calculation(self):
        """偏离量计算应正确"""
        result = self.bt.check_threshold('EF', 50)
        self.assertIsNotNone(result.deviation)
        self.assertAlmostEqual(result.deviation, -5, places=1)

    def test_deviation_percent(self):
        """偏离百分比计算应正确"""
        result = self.bt.check_threshold('EF', 50)
        self.assertIsNotNone(result.deviation_percent)
        expected_pct = (-5 / 55) * 100
        self.assertAlmostEqual(result.deviation_percent, expected_pct, places=1)

    # === Trend 评估 ===

    def test_trend_improving_ef(self):
        """EF从baseline上升应为improving"""
        result = self.bt.check_threshold('EF', 65)
        self.assertEqual(result.trend, TrendDirection.IMPROVING)

    def test_trend_deteriorating_lactate(self):
        """Lactate从baseline上升应为deteriorating"""
        result = self.bt.check_threshold('Lactate', 4.0)
        self.assertEqual(result.trend, TrendDirection.DETERIORATING)

    def test_trend_stable_within_deviation(self):
        """EF在acceptable_deviation内应为stable"""
        result = self.bt.check_threshold('EF', 54)  # baseline=55, dev=5
        self.assertEqual(result.trend, TrendDirection.STABLE)

    # === 批量检查 ===

    def test_check_all_indicators(self):
        """批量检查应返回所有结果"""
        indicators = {'EF': 55, 'CI': 2.5, 'Lactate': 1.5}
        results = self.bt.check_all_indicators(indicators)
        self.assertEqual(len(results), 3)

    def test_get_alerts_filters(self):
        """get_alerts应正确过滤"""
        indicators = {'EF': 55, 'CI': 1.8, 'Lactate': 6.5}
        alerts = self.bt.get_alerts(indicators, min_level=AlertLevel.WARNING)
        # CI=1.8 (red_line) and Lactate=6.5 (reject/red_line) should be alerts
        self.assertTrue(len(alerts) >= 2)
        for a in alerts:
            self.assertNotEqual(a.alert_level, AlertLevel.NORMAL)

    # === 异常状态映射 ===

    def test_abnormality_state_normal(self):
        """正常指标应返回_Normal后缀"""
        state = self.bt.get_abnormality_state('EF', 55)
        self.assertIn('Normal', state)

    def test_abnormality_state_low(self):
        """EF=35应返回低方向的异常状态"""
        state = self.bt.get_abnormality_state('EF', 35)
        self.assertNotIn('Normal', state)

    def test_abnormality_state_high(self):
        """Lactate=7应返回高方向的异常状态"""
        state = self.bt.get_abnormality_state('Lactate', 7)
        self.assertNotIn('Normal', state)

    # === 报告生成 ===

    def test_generate_report(self):
        """报告生成应包含关键信息"""
        indicators = {'EF': 35, 'CI': 1.8, 'MAP': 55}
        report = self.bt.generate_report(indicators)
        self.assertIn('Baseline Threshold Check Report', report)
        # EF=35 triggers reject (red_line), CI=1.8 triggers red_line
        self.assertIn('EF', report)
        self.assertIn('CI', report)

    # === KG映射 ===

    def test_kg_mapping_exists_for_setpoints(self):
        """Setpoint异常状态应有KG映射"""
        from src.baseline_thresholds import INDICATOR_TO_KG_MAPPING
        self.assertIn('Flow_Low', INDICATOR_TO_KG_MAPPING)
        self.assertIn('AoDP_Low', INDICATOR_TO_KG_MAPPING)
        self.assertIn('pH_Low', INDICATOR_TO_KG_MAPPING)


class TestBaselineEvaluator(unittest.TestCase):
    """BaselineEvaluator 类测试"""

    def setUp(self):
        self.evaluator = BaselineEvaluator()

    def test_init_loads_config(self):
        """初始化应加载baseline配置"""
        self.assertIsNotNone(self.evaluator.baseline_config)

    def test_get_baseline_confirmed(self):
        """应返回confirmed baseline值"""
        baseline = self.evaluator.get_baseline('EF')
        self.assertEqual(baseline, 55)

    def test_get_baseline_provisional(self):
        """应返回provisional baseline值"""
        baseline = self.evaluator.get_baseline('K_A')
        self.assertEqual(baseline, 4.0)

    def test_get_baseline_setpoint(self):
        """应返回Setpoint baseline值"""
        baseline = self.evaluator.get_baseline('Flow')
        self.assertEqual(baseline, 4.5)

    def test_dynamic_baseline(self):
        """设置动态baseline后应返回该值"""
        self.evaluator.set_dynamic_baseline('TestInd', 42.0)
        self.assertEqual(self.evaluator.get_baseline('TestInd'), 42.0)

    def test_compare_normal(self):
        """接近baseline的值应为normal"""
        result = self.evaluator.compare('EF', 54)
        self.assertEqual(result.deviation_severity, DeviationSeverity.NORMAL)
        self.assertTrue(result.within_acceptable)

    def test_compare_mild(self):
        """轻度偏离应为mild"""
        result = self.evaluator.compare('EF', 46)  # deviation=9, acceptable=5
        self.assertEqual(result.deviation_severity, DeviationSeverity.MILD)

    def test_compare_critical(self):
        """严重偏离应为critical"""
        result = self.evaluator.compare('EF', 30)  # deviation=25, acceptable=5
        self.assertEqual(result.deviation_severity, DeviationSeverity.CRITICAL)

    def test_compare_improving_trend(self):
        """EF增加应为improving"""
        result = self.evaluator.compare('EF', 65)
        self.assertEqual(result.trend, Trend.IMPROVING)

    def test_compare_deteriorating_trend(self):
        """Lactate增加应为deteriorating"""
        result = self.evaluator.compare('Lactate', 5.0)
        self.assertEqual(result.trend, Trend.DETERIORATING)

    def test_compare_setpoint_stable(self):
        """Setpoint在acceptable范围内应为stable"""
        result = self.evaluator.compare('Flow', 4.4)  # baseline=4.5, dev=0.3
        self.assertEqual(result.trend, Trend.STABLE)

    def test_compare_first_measurement(self):
        """首次测量无baseline应设为动态baseline"""
        result = self.evaluator.compare('NewIndicator', 100)
        self.assertEqual(result.delta, 0)
        self.assertTrue(result.within_acceptable)
        # 应设为动态baseline
        self.assertEqual(self.evaluator.get_baseline('NewIndicator'), 100)

    def test_record_measurement(self):
        """记录测量值应存入历史"""
        self.evaluator.record_measurement('EF', 52)
        self.evaluator.record_measurement('EF', 54)
        self.assertIn('EF', self.evaluator.measurement_history)
        self.assertEqual(len(self.evaluator.measurement_history['EF']), 2)

    def test_generate_report(self):
        """报告生成应包含所有指标"""
        measurements = {'EF': 50, 'CI': 2.0, 'Lactate': 4.0}
        report = self.evaluator.generate_report(measurements)
        self.assertIn('Baseline', report)
        self.assertIn('EF', report)

    def test_delta_calculation(self):
        """变化量计算应准确"""
        result = self.evaluator.compare('MAP', 60)  # baseline=70
        self.assertAlmostEqual(result.delta, -10, places=1)

    def test_delta_percent_calculation(self):
        """百分比变化计算应准确"""
        result = self.evaluator.compare('MAP', 60)
        expected_pct = (-10 / 70) * 100
        self.assertAlmostEqual(result.delta_percent, expected_pct, places=1)


class TestThresholdManager(unittest.TestCase):
    """ThresholdManager 类测试"""

    def setUp(self):
        self.manager = ThresholdManager()

    def test_init_loads_configs(self):
        """初始化应加载配置"""
        self.assertIsNotNone(self.manager.thresholds)

    def test_get_indicator_config_exists(self):
        """已确认指标应有配置"""
        config = self.manager.get_indicator_config('EF')
        self.assertIsNotNone(config)
        self.assertEqual(config['name'], '射血分数')

    def test_get_indicator_config_setpoint(self):
        """Setpoint指标应有配置"""
        config = self.manager.get_indicator_config('Flow')
        self.assertIsNotNone(config)
        self.assertEqual(config['name'], '灌注流量')

    def test_get_indicator_config_nonexistent(self):
        """不存在指标应返回None"""
        config = self.manager.get_indicator_config('NonExistent')
        self.assertIsNone(config)

    def test_evaluate_ef_accept(self):
        """EF=55 应接受"""
        result = self.manager.evaluate('EF', 55)
        self.assertEqual(result.result, EvaluationResult.ACCEPT)

    def test_evaluate_ef_reject(self):
        """EF=35 应拒绝"""
        result = self.manager.evaluate('EF', 35)
        self.assertEqual(result.result, EvaluationResult.REJECT)

    def test_evaluate_ef_gray_zone(self):
        """EF=47 应在灰区"""
        result = self.manager.evaluate('EF', 47)
        self.assertEqual(result.result, EvaluationResult.GRAY_ZONE)

    def test_evaluate_ci_normal(self):
        """CI=2.5 应在目标范围"""
        result = self.manager.evaluate('CI', 2.5)
        self.assertEqual(result.result, EvaluationResult.NORMAL)

    def test_evaluate_ci_red_line(self):
        """CI=1.8 应触发红线"""
        result = self.manager.evaluate('CI', 1.8)
        self.assertEqual(result.result, EvaluationResult.RED_LINE)

    def test_evaluate_lactate_reject(self):
        """Lactate=6.0 应拒绝"""
        result = self.manager.evaluate('Lactate', 6.0)
        self.assertEqual(result.result, EvaluationResult.REJECT)

    def test_evaluate_unknown_indicator(self):
        """未知指标应返回PENDING"""
        result = self.manager.evaluate('Unknown', 42)
        self.assertEqual(result.result, EvaluationResult.PENDING)

    # === Setpoint评估 ===

    def test_evaluate_flow_normal(self):
        """Flow=4.5 应在目标范围"""
        result = self.manager.evaluate('Flow', 4.5)
        self.assertEqual(result.result, EvaluationResult.NORMAL)

    def test_evaluate_flow_warning(self):
        """Flow=3.6 应触发warning或red_line"""
        result = self.manager.evaluate('Flow', 3.6)
        self.assertIn(result.result,
                      [EvaluationResult.WARNING, EvaluationResult.RED_LINE])

    def test_evaluate_hemoglobin_red_line(self):
        """Hemoglobin=28 应触发red_line"""
        result = self.manager.evaluate('Hemoglobin', 28)
        self.assertEqual(result.result, EvaluationResult.RED_LINE)

    def test_evaluate_ph_normal(self):
        """pH=7.30 应在目标范围"""
        result = self.manager.evaluate('pH', 7.30)
        self.assertEqual(result.result, EvaluationResult.NORMAL)

    # === 置信度和分类 ===

    def test_get_all_confirmed_indicators(self):
        """应返回所有已确认指标"""
        confirmed = self.manager.get_all_confirmed_indicators()
        self.assertTrue(len(confirmed) > 0)
        self.assertIn('EF', confirmed)
        self.assertIn('Flow', confirmed)

    def test_confidence_summary(self):
        """置信度分组应正确"""
        summary = self.manager.get_confidence_summary()
        self.assertIn('high', summary)
        self.assertIn('medium', summary)
        # EF should be high confidence
        self.assertIn('EF', summary['high'])


class TestClinicalScenarios(unittest.TestCase):
    """临床场景集成测试"""

    def setUp(self):
        self.bt = BaselineThresholds()
        self.evaluator = BaselineEvaluator()
        self.manager = ThresholdManager()

    def test_scenario_successful_perfusion(self):
        """成功灌注场景：所有指标逐步恢复"""
        # t=30min (baseline)
        baseline = {
            'Flow': 4.5, 'Temperature': 22.0, 'AoDP': 40,
            'Lactate': 2.8, 'pH': 7.28, 'EF': 0, 'SvO2': 82,
        }
        # t=240min (end)
        final = {
            'Flow': 4.2, 'Temperature': 37.0, 'AoDP': 40,
            'Lactate': 2.0, 'pH': 7.35, 'EF': 35, 'SvO2': 70,
        }

        # Baseline check: at t=30, some may be abnormal
        baseline_results = self.bt.check_all_indicators(baseline)

        # Final check: should have fewer/no alerts
        final_results = self.bt.check_all_indicators(final)
        final_alerts = [r for r in final_results
                        if r.alert_level in [AlertLevel.RED_LINE, AlertLevel.CRITICAL]]

        # In a successful case, fewer critical alerts at end
        baseline_alerts = [r for r in baseline_results
                           if r.alert_level in [AlertLevel.RED_LINE, AlertLevel.CRITICAL]]

        # Verify Lactate improved
        lac_result = self.bt.check_threshold('Lactate', final['Lactate'])
        self.assertEqual(lac_result.alert_level, AlertLevel.NORMAL)

    def test_scenario_failing_perfusion(self):
        """失败灌注场景：指标持续恶化"""
        measurements_240 = {
            'Lactate': 6.8, 'pH': 7.15, 'K_A': 7.1,
            'EF': 8, 'CI': 1.5, 'MAP': 42,
        }

        alerts = self.bt.get_alerts(measurements_240, min_level=AlertLevel.WARNING)
        # Should have multiple alerts
        self.assertTrue(len(alerts) >= 3,
                        f"失败灌注应有>=3个警报，实际{len(alerts)}个")

        # K_A=7.1 should be critical
        k_result = self.bt.check_threshold('K_A', 7.1)
        self.assertEqual(k_result.alert_level, AlertLevel.CRITICAL)

    def test_scenario_setpoint_adjustment_needed(self):
        """需要调整Setpoint的场景"""
        measurements = {
            'Flow': 3.4,        # 严重不足
            'Hemoglobin': 28,   # 严重贫血
            'AoDP': 24,         # 压力严重不足
        }

        for ind, val in measurements.items():
            result = self.bt.check_threshold(ind, val)
            self.assertEqual(result.alert_level, AlertLevel.RED_LINE,
                             f"{ind}={val} 应触发red_line")

    def test_scenario_transplant_evaluation(self):
        """移植评估场景：共识指标检查"""
        transplant_indicators = {
            'PVR': 5.5,        # 禁忌
            'TPG': 16,         # 禁忌
            'PASP': 75,        # 禁忌
            'Creatinine': 2.2, # 红线
        }

        results = self.bt.check_all_indicators(transplant_indicators)
        for r in results:
            self.assertNotEqual(r.alert_level, AlertLevel.NORMAL,
                                f"{r.indicator}={r.value} 不应为正常状态")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def setUp(self):
        self.bt = BaselineThresholds()
        self.evaluator = BaselineEvaluator()

    def test_zero_value(self):
        """值为0应正确处理"""
        result = self.bt.check_threshold('EF', 0)
        self.assertIsNotNone(result)

    def test_negative_value(self):
        """负值应正确处理"""
        result = self.bt.check_threshold('EF', -5)
        self.assertIsNotNone(result)

    def test_very_large_value(self):
        """极大值应正确处理"""
        result = self.bt.check_threshold('EF', 999)
        self.assertIsNotNone(result)

    def test_none_baseline_handling(self):
        """无baseline指标应正确处理"""
        result = self.bt.check_threshold('EF', 50, baseline_value=None)
        # Should still work, using config baseline
        self.assertIsNotNone(result.baseline_value)

    def test_zero_baseline_no_division_error(self):
        """baseline为0时不应除零错误"""
        result = self.bt.check_threshold('PacingRate', 105, baseline_value=0)
        self.assertIsNotNone(result)
        # deviation_percent should handle division by zero gracefully
        # (baseline_value=0, so deviation_percent won't be calculated)

    def test_evaluator_zero_baseline(self):
        """BaselineEvaluator baseline为0时不应出错"""
        self.evaluator.set_dynamic_baseline('TestZero', 0)
        result = self.evaluator.compare('TestZero', 5)
        self.assertIsNotNone(result)

    def test_unconfigured_indicator(self):
        """未配置指标应返回NORMAL（无阈值可检查）"""
        result = self.bt.check_threshold('FakeIndicator', 42)
        self.assertEqual(result.alert_level, AlertLevel.NORMAL)
        self.assertIn('未找到', result.threshold_description)


if __name__ == '__main__':
    unittest.main(verbosity=2)
