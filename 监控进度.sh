#!/bin/bash
# 知识图谱构建进度监控脚本

echo "============================================================"
echo "心脏移植知识图谱构建 - 实时监控"
echo "============================================================"
echo ""

# 检查进程
PID=$(cat /tmp/extraction_pid.txt 2>/dev/null)
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ 抽取进程运行中 (PID: $PID)"
else
    echo "✗ 抽取进程未运行"
    exit 1
fi

echo ""
echo "------------------------------------------------------------"
echo "进度统计"
echo "------------------------------------------------------------"

# 统计已处理文章
PROCESSED=$(ls /home/user/SHU/cache/parsed_triples/ 2>/dev/null | wc -l)
TOTAL=24432
PROGRESS=$(awk "BEGIN {printf \"%.2f\", ($PROCESSED/$TOTAL)*100}")

echo "已处理文章: $PROCESSED / $TOTAL ($PROGRESS%)"

# 计算速度和剩余时间
if [ -f /tmp/extraction_start_time ]; then
    START_TIME=$(cat /tmp/extraction_start_time)
else
    date +%s > /tmp/extraction_start_time
    START_TIME=$(cat /tmp/extraction_start_time)
fi

CURRENT_TIME=$(date +%s)
ELAPSED=$((CURRENT_TIME - START_TIME))

if [ $ELAPSED -gt 0 ] && [ $PROCESSED -gt 0 ]; then
    SPEED=$(awk "BEGIN {printf \"%.2f\", $PROCESSED/$ELAPSED}")
    REMAINING=$((TOTAL - PROCESSED))
    ETA_SECONDS=$(awk "BEGIN {printf \"%.0f\", $REMAINING/$SPEED}")
    ETA_HOURS=$(awk "BEGIN {printf \"%.1f\", $ETA_SECONDS/3600}")

    echo "处理速度: $SPEED 篇/秒"
    echo "预计剩余时间: $ETA_HOURS 小时"
fi

echo ""
echo "------------------------------------------------------------"
echo "最新日志 (最后10行)"
echo "------------------------------------------------------------"
tail -10 /home/user/SHU/logs/full_extraction.log

echo ""
echo "------------------------------------------------------------"
echo "缓存统计"
echo "------------------------------------------------------------"
RAW_COUNT=$(ls /home/user/SHU/cache/llm_raw_outputs/ 2>/dev/null | wc -l)
PARSED_COUNT=$(ls /home/user/SHU/cache/parsed_triples/ 2>/dev/null | wc -l)

echo "原始LLM输出: $RAW_COUNT 个文件"
echo "解析后三元组: $PARSED_COUNT 个文件"

echo ""
echo "============================================================"
echo "监控命令："
echo "  实时日志: tail -f /home/user/SHU/logs/full_extraction.log"
echo "  重新监控: bash /home/user/SHU/监控进度.sh"
echo "============================================================"
