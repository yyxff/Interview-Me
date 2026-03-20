#!/usr/bin/env bash
# ============================================================
# Interview Me - 一键启动脚本
# 用法: ./start.sh [--api-key sk-xxx]
#
# 日志:
#   logs/backend.log  - FastAPI / uvicorn 输出
#   logs/frontend.log - Vite 开发服务器输出
#
# 热更新说明:
#   - 前端: Vite HMR，保存文件后浏览器自动更新，无需重启
#   - 后端: uvicorn --reload，保存 .py 文件后自动重载，无需重启
# ============================================================

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$ROOT/logs"

# ---- 加载 backend/.env ----
ENV_FILE="$ROOT/backend/.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --api-key) LLM_API_KEY="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

# ---- 颜色 ----
GRN='\033[0;32m'; BLU='\033[0;34m'; YLW='\033[1;33m'
RED='\033[0;31m'; DIM='\033[2m'; RST='\033[0m'

# ---- Banner ----
echo -e "${BLU}"
echo "  ╔══════════════════════════════════╗"
echo "  ║     Interview Me  模拟面试        ║"
echo "  ╚══════════════════════════════════╝"
echo -e "${RST}"

# ---- 准备日志目录 ----
mkdir -p "$LOGS"
BACKEND_LOG="$LOGS/backend.log"
FRONTEND_LOG="$LOGS/frontend.log"

# 每次启动在日志头写入分隔线和时间戳
{
  echo ""
  echo "════════════════════════════════════════"
  echo "  启动时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "════════════════════════════════════════"
} | tee -a "$BACKEND_LOG" >> "$FRONTEND_LOG"

# ---- 检查 conda 环境 ----
if ! conda env list 2>/dev/null | grep -q "^interview-me "; then
  echo -e "${RED}✗ conda 环境 'interview-me' 不存在，请先执行:${RST}"
  echo "    conda create -n interview-me python=3.11 -y"
  echo "    conda run -n interview-me pip install -r backend/requirements.txt"
  exit 1
fi

# ---- 释放端口（清理残留进程）----
free_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo -e "${DIM}  清理端口 $port 上的旧进程 ($pids)${RST}"
    kill -9 $pids 2>/dev/null || true
    sleep 0.3
  fi
}

free_port 8000
free_port 3000

# ---- 启动后端 ----
echo -e "${YLW}▶  启动后端 (FastAPI :8000)...${RST}"

(
  cd "$ROOT/backend"
  # 环境变量已由 .env 加载，此处无需额外处理
  conda run --no-capture-output -n interview-me \
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
) >> "$BACKEND_LOG" 2>&1 &

BACKEND_PID=$!

# 等待后端就绪（最多 12s）
printf "   等待后端就绪"
READY=0
for _ in {1..24}; do
  sleep 0.5
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    READY=1; break
  fi
  printf "."
done

if [ "$READY" -eq 1 ]; then
  echo -e " ${GRN}OK${RST}"
  echo -e "   ${GRN}✓${RST} Backend PID: $BACKEND_PID"
else
  echo -e " ${RED}超时${RST}"
  echo -e "   ${RED}✗${RST} 后端可能启动失败，查看: ${DIM}$BACKEND_LOG${RST}"
fi

# ---- 启动前端 ----
echo -e "${YLW}▶  启动前端 (Vite :3000)...${RST}"

(
  cd "$ROOT/frontend"
  npm run dev
) >> "$FRONTEND_LOG" 2>&1 &

FRONTEND_PID=$!

# 等待 Vite 就绪
printf "   等待前端就绪"
READY=0
for _ in {1..20}; do
  sleep 0.5
  if curl -sf http://localhost:3000 > /dev/null 2>&1; then
    READY=1; break
  fi
  printf "."
done

if [ "$READY" -eq 1 ]; then
  echo -e " ${GRN}OK${RST}"
  echo -e "   ${GRN}✓${RST} Frontend PID: $FRONTEND_PID"
else
  echo -e " ${YLW}未检测到（Vite 可能仍在启动）${RST}"
  echo -e "   Frontend PID: $FRONTEND_PID"
fi

# ---- 摘要 ----
echo ""
echo -e "  前端    ${GRN}http://localhost:3000${RST}"
echo -e "  后端    ${GRN}http://localhost:8000${RST}"
echo -e "  API文档 ${GRN}http://localhost:8000/docs${RST}"
if [ -n "${LLM_API_KEY:-}" ] || [ "${LLM_PROVIDER:-}" = "openai-compatible" ]; then
  echo -e "  AI      ${GRN}已启用 (${LLM_PROVIDER:-anthropic} / ${LLM_MODEL:-默认模型})${RST}"
else
  echo -e "  AI      ${YLW}未配置 API Key（占位模式）${RST}"
  echo -e "  ${DIM}提示: 在 backend/.env 中填写 LLM_API_KEY${RST}"
fi
echo ""
echo -e "  日志  ${DIM}tail -f $BACKEND_LOG${RST}"
echo -e "        ${DIM}tail -f $FRONTEND_LOG${RST}"
echo ""
echo -e "${YLW}  Ctrl+C 停止所有服务${RST}"
echo ""

# ---- 清理函数 ----
cleanup() {
  echo ""
  echo -e "${YLW}正在停止服务...${RST}"
  # 杀掉进程组，确保 uvicorn reload 的子进程也被清理
  kill -- -$BACKEND_PID  2>/dev/null || kill $BACKEND_PID  2>/dev/null || true
  kill -- -$FRONTEND_PID 2>/dev/null || kill $FRONTEND_PID 2>/dev/null || true
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  echo -e "${GRN}已停止。日志保留在 logs/ 目录。${RST}"
  exit 0
}

trap cleanup INT TERM

# 主进程保持运行，等待子进程或 Ctrl+C
wait
