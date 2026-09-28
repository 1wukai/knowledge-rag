#!/usr/bin/env bash
# 一条命令启动：安装依赖 → 检查前端 → local 模式启动（首启自动索引内置语料）。
# 全部数据落在可丢弃的 data/ 目录；./run.sh --reset 清空索引与运行时设置。
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--reset" ]]; then
  rm -rf data/qdrant_local data/index_state.json data/runtime_settings.json
  echo "已清空 data/ 下可丢弃数据（索引、运行时设置）。"
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  PY=(uv run python)
  uv sync
else
  echo "未检测到 uv，回退 venv + pip（需 Python 3.12）。"
  PYTHON_BIN="$(command -v python3.12 || command -v python3)"
  [[ -d .venv ]] || "$PYTHON_BIN" -m venv .venv
  .venv/bin/python -m pip install -q -e .
  PY=(.venv/bin/python)
fi

if [[ ! -d web/dist ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "前端未构建，执行 npm ci && npm run build（仅首次需要）..."
    (cd web && npm ci && npm run build)
  else
    echo "错误：缺少 web/dist 且无 npm，请参考 README 构建前端。" >&2
    exit 1
  fi
fi

echo "local 模式启动：内嵌索引 + 默认 mock 模型；控制台 http://127.0.0.1:8000"
echo "使用真实模型：控制台「设置」页填写 base_url / model / key 后保存并测试连通性。"
exec "${PY[@]}" scripts/serve.py --local --mock
