#!/bin/bash

# 停止supervisord
echo "Stopping supervisord..."

# 使用配置文件执行supervisorctl
supervisorctl -c supervisord.conf shutdown

# 如果上面的命令失败，尝试直接杀死进程
if [ -f logs/supervisord.pid ]; then
    echo "Killing process using PID file..."
    PID=$(cat logs/supervisord.pid)
    kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
    rm logs/supervisord.pid
fi

# 清理其他相关文件
rm -f logs/supervisor.sock 2>/dev/null

# 查找并杀死所有uvicorn进程
echo "Cleaning up uvicorn processes..."
pkill -f "uvicorn main:app"

echo "Service stopped" 