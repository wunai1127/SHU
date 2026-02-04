#!/usr/bin/env python3
"""
教育智能体速成平台 - 启动脚本
==============================

使用方式:
    python run_web.py

或直接使用 streamlit:
    streamlit run web/app.py --server.port 8001
"""

import os
import sys
from pathlib import Path

# 确保在正确的目录
os.chdir(Path(__file__).parent)

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies():
    """检查依赖"""
    missing = []

    try:
        import streamlit
    except ImportError:
        missing.append("streamlit")

    try:
        import openai
    except ImportError:
        missing.append("openai")

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False

    return True


def check_env():
    """检查环境变量"""
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("=" * 50)
        print("⚠️  警告: API Key 未配置")
        print("=" * 50)
        print()
        print("请设置以下环境变量:")
        print()
        print("  # Linux/Mac:")
        print("  export LLM_API_KEY='your-deepseek-api-key'")
        print("  export LLM_BASE_URL='https://api.deepseek.com/v1'")
        print()
        print("  # Windows PowerShell:")
        print("  $env:LLM_API_KEY='your-deepseek-api-key'")
        print("  $env:LLM_BASE_URL='https://api.deepseek.com/v1'")
        print()
        print("  # 或在 .env 文件中配置")
        print()
        print("服务将以演示模式启动（无法实际调用AI）...")
        print("=" * 50)
        print()


def main():
    """启动服务"""
    print("=" * 50)
    print("🎓 教育智能体速成平台")
    print("=" * 50)
    print()

    if not check_dependencies():
        sys.exit(1)

    check_env()

    port = int(os.getenv('WEB_PORT', '8001'))

    print(f"启动服务中...")
    print(f"访问地址: http://localhost:{port}")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 50)

    # 启动 Streamlit
    os.system(f'streamlit run web/app.py --server.port {port} --server.address 0.0.0.0')


if __name__ == "__main__":
    main()
