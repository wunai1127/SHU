#!/bin/bash
# 抽取进程守护脚本 - 自动检测并重启中断的进程
# 用途：代理不稳定导致进程经常中断，此脚本自动重启

LOG_FILE="/home/user/SHU/logs/extract_from_6000.log"
CHECKPOINT="/home/user/SHU/cache/extract_6000_checkpoint.json"
SCRIPT="/home/user/SHU/automated_kg_pipeline/extract_from_6000.py"
DAEMON_LOG="/home/user/SHU/logs/extraction_daemon.log"
PID_FILE="/home/user/SHU/logs/extract_6000.pid"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DAEMON_LOG"
}

# 检查进程是否在运行
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            # 检查进程是否卡住（超过5分钟无输出）
            if [ -f "$LOG_FILE" ]; then
                LAST_LOG_TIME=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo 0)
                CURRENT_TIME=$(date +%s)
                TIME_DIFF=$((CURRENT_TIME - LAST_LOG_TIME))

                if [ $TIME_DIFF -gt 300 ]; then
                    log "⚠️  进程存在但已卡住 ${TIME_DIFF}秒，准备重启"
                    kill -9 "$PID" 2>/dev/null
                    return 1
                fi
            fi
            return 0
        fi
    fi
    return 1
}

# 启动抽取进程
start_extraction() {
    log "🚀 启动抽取进程..."
    cd /home/user/SHU
    nohup python3 -u "$SCRIPT" > "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    log "✅ 进程已启动: PID $NEW_PID"

    # 等待3秒检查是否成功启动
    sleep 3
    if ps -p "$NEW_PID" > /dev/null 2>&1; then
        log "✅ 进程确认运行中"
        # 显示当前进度
        if [ -f "$CHECKPOINT" ]; then
            PROCESSED=$(python3 -c "import json; print(len(json.load(open('$CHECKPOINT'))['processed_ids']))" 2>/dev/null || echo "0")
            log "📊 已处理: $PROCESSED/6216 篇"
        fi
    else
        log "❌ 进程启动失败"
    fi
}

# 主循环
log "=========================================="
log "抽取进程守护脚本启动"
log "监控脚本: $SCRIPT"
log "检查间隔: 30秒"
log "=========================================="

while true; do
    if ! is_running; then
        log "⚠️  检测到进程已停止"
        start_extraction
    fi

    # 每30秒检查一次
    sleep 30
done
