#!/usr/bin/env python3
"""
系统就绪性检查脚本
运行此脚本验证所有依赖和配置是否正确
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

class SystemReadinessChecker:
    """系统就绪性检查器"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.issues = []
        self.warnings = []

    def check_all(self) -> bool:
        """执行所有检查"""
        print("=" * 60)
        print("心脏移植KG系统就绪性检查")
        print("=" * 60)
        print()

        checks = [
            ("Python版本", self.check_python_version),
            ("配置文件", self.check_config_file),
            ("依赖包", self.check_dependencies),
            ("数据文件", self.check_data_files),
            ("Schema文件", self.check_schema_file),
            ("Neo4j连接", self.check_neo4j_connection),
            ("LLM配置", self.check_llm_config),
            ("GPU可用性", self.check_gpu_availability),
        ]

        for name, check_func in checks:
            print(f"检查 {name}...", end=" ")
            try:
                result = check_func()
                if result:
                    print("✓")
                else:
                    print("✗")
            except Exception as e:
                print(f"✗ (错误: {e})")
                self.issues.append(f"{name}: {e}")

        print()
        print("=" * 60)
        self.print_summary()
        print("=" * 60)

        return len(self.issues) == 0

    def check_python_version(self) -> bool:
        """检查Python版本"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 10:
            return True
        else:
            self.issues.append(f"Python版本过低: {version.major}.{version.minor} (需要3.10+)")
            return False

    def check_config_file(self) -> bool:
        """检查配置文件"""
        if not Path(self.config_path).exists():
            self.issues.append(f"配置文件不存在: {self.config_path}")
            self.warnings.append("提示: cp config_template.yaml config.yaml")
            return False

        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except Exception as e:
            self.issues.append(f"配置文件格式错误: {e}")
            return False

    def check_dependencies(self) -> bool:
        """检查依赖包"""
        required_packages = [
            "yaml",
            "tqdm",
            "openai",
            "neo4j",
            "pandas",
            "numpy"
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            self.issues.append(f"缺少依赖包: {', '.join(missing)}")
            self.warnings.append("运行: pip install -r requirements.txt")
            return False

        return True

    def check_data_files(self) -> bool:
        """检查数据文件"""
        if not hasattr(self, 'config'):
            self.warnings.append("跳过数据文件检查（配置文件未加载）")
            return True

        data_dir = Path(self.config['data']['input_directory'])
        if not data_dir.exists():
            self.issues.append(f"数据目录不存在: {data_dir}")
            return False

        pattern = self.config['data']['filename_pattern']
        files = list(data_dir.glob(pattern))

        if len(files) == 0:
            self.issues.append(f"数据目录为空: {data_dir}/{pattern}")
            return False

        print(f"({len(files)} 文件)", end=" ")
        return True

    def check_schema_file(self) -> bool:
        """检查Schema文件"""
        if not hasattr(self, 'config'):
            self.warnings.append("跳过Schema检查（配置文件未加载）")
            return True

        schema_file = Path(self.config['schema']['schema_file'])
        if not schema_file.exists():
            self.issues.append(f"Schema文件不存在: {schema_file}")
            return False

        return True

    def check_neo4j_connection(self) -> bool:
        """检查Neo4j连接"""
        if not hasattr(self, 'config'):
            self.warnings.append("跳过Neo4j检查（配置文件未加载）")
            return True

        try:
            from neo4j import GraphDatabase

            neo4j_config = self.config['neo4j']
            driver = GraphDatabase.driver(
                neo4j_config['uri'],
                auth=(neo4j_config['username'], neo4j_config['password'])
            )

            with driver.session() as session:
                result = session.run("RETURN 1 AS test")
                assert result.single()['test'] == 1

            driver.close()
            return True

        except Exception as e:
            self.issues.append(f"Neo4j连接失败: {e}")
            self.warnings.append("检查Neo4j是否运行: sudo systemctl status neo4j")
            return False

    def check_llm_config(self) -> bool:
        """检查LLM配置"""
        if not hasattr(self, 'config'):
            self.warnings.append("跳过LLM检查（配置文件未加载）")
            return True

        provider = self.config['llm']['provider']

        if provider == "openai":
            api_key = self.config['llm']['openai'].get('api_key', '')
            if api_key == "YOUR_OPENAI_API_KEY" or not api_key:
                self.issues.append("OpenAI API密钥未配置")
                return False

        elif provider == "deepseek":
            api_key = self.config['llm']['deepseek'].get('api_key', '')
            if api_key == "YOUR_DEEPSEEK_API_KEY" or not api_key:
                self.issues.append("DeepSeek API密钥未配置")
                return False

        elif provider == "local":
            model_path = self.config['llm']['local']['model_path']
            print(f"(模型: {model_path})", end=" ")
            # 本地模型暂不验证是否下载

        return True

    def check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"({num_gpus} GPU)", end=" ")
                return True
            else:
                self.warnings.append("未检测到GPU，将使用CPU（速度较慢）")
                return True
        except ImportError:
            self.warnings.append("PyTorch未安装，无法检测GPU")
            return True

    def print_summary(self):
        """打印检查摘要"""
        if self.issues:
            print("\n❌ 发现以下问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"  {i}. {issue}")

        if self.warnings:
            print("\n⚠️  警告:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if not self.issues:
            print("\n✅ 所有检查通过！系统已就绪。")
            print("\n运行以下命令开始构建知识图谱:")
            print("  python auto_kg_builder.py --config config.yaml")
        else:
            print(f"\n❌ 发现 {len(self.issues)} 个问题，请先解决后再运行。")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='检查系统是否就绪')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    args = parser.parse_args()

    checker = SystemReadinessChecker(args.config)
    success = checker.check_all()

    sys.exit(0 if success else 1)
