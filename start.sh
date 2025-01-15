#!/bin/bash

# 加载环境变量
set -a
source .env
set +a

# 创建logs目录（如果不存在）
mkdir -p logs

# 检查supervisord是否已经运行
if [ -f logs/supervisord.pid ]; then
    echo "Stopping existing supervisord..."
    supervisorctl shutdown
    sleep 2
fi

# 启动supervisord
echo "Starting supervisord..."
supervisord -c supervisord.conf

# 显示进程状态
echo "Showing process status..."
supervisorctl status

# 监控日志
echo "Tailing logs..."
tail -f logs/fastapi.out.log 