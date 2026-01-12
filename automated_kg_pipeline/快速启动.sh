#!/bin/bash
# 心脏移植KG构建系统 - 快速启动脚本

set -e  # 遇到错误立即退出

echo "=========================================="
echo "心脏移植知识图谱自动化构建系统"
echo "=========================================="
echo ""

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo "❌ 配置文件不存在！"
    echo ""
    echo "请先完成配置："
    echo "1. cp config_template.yaml config.yaml"
    echo "2. 编辑 config.yaml，填写 API 密钥和 Neo4j 密码"
    echo "3. 阅读 配置指南.md 了解详细步骤"
    exit 1
fi

echo "✓ 配置文件存在"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python版本: $PYTHON_VERSION"

# 检查依赖
echo ""
echo "检查依赖包..."
if ! python3 -c "import yaml, tqdm, openai, neo4j, pandas" 2>/dev/null; then
    echo "⚠️  缺少依赖包，正在安装..."
    pip install -r requirements.txt
    echo "✓ 依赖包安装完成"
else
    echo "✓ 依赖包已安装"
fi

# 运行系统检查
echo ""
echo "=========================================="
echo "运行系统就绪性检查..."
echo "=========================================="
python3 test_system_ready.py --config config.yaml

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 系统检查未通过，请先解决上述问题"
    echo ""
    echo "常见问题排查："
    echo "1. 检查 config.yaml 中的 API 密钥是否填写"
    echo "2. 检查 Neo4j 是否运行: sudo systemctl status neo4j"
    echo "3. 检查数据文件是否上传到 data/medical_abstracts/"
    echo ""
    echo "详细帮助: 阅读 配置指南.md"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ 系统检查通过！"
echo "=========================================="
echo ""

# 询问用户是否继续
read -p "是否开始构建知识图谱？(y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 询问模式
echo ""
echo "选择运行模式："
echo "1. 测试模式（仅处理10篇文章，快速验证）"
echo "2. 全量模式（处理全部20000+篇文章）"
read -p "请选择 [1/2]: " MODE

if [ "$MODE" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "启动测试模式（10篇文章）"
    echo "=========================================="
    python3 auto_kg_builder.py --config config.yaml --test-mode --max-articles 10
else
    echo ""
    echo "=========================================="
    echo "启动全量模式（全部文章）"
    echo "=========================================="
    echo "预计时间："
    echo "  - DeepSeek API: 30-60分钟"
    echo "  - 本地4 GPU: 约14小时"
    echo "=========================================="
    python3 auto_kg_builder.py --config config.yaml
fi

# 检查是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 知识图谱构建完成！"
    echo "=========================================="
    echo ""
    echo "查看结果："
    echo "1. 构建报告: output/final/build_report.json"
    echo "2. Neo4j浏览器: http://localhost:7474"
    echo "3. 日志文件: logs/kg_construction.log"
    echo ""
    echo "下一步："
    echo "1. 在Neo4j中验证数据质量"
    echo "2. 查看人工验证样本: validation/manual_review_samples.json"
    echo "3. 开始集成Multi-Agent系统"
else
    echo ""
    echo "❌ 构建过程出错，请查看日志："
    echo "   tail -100 logs/kg_construction.log"
fi
