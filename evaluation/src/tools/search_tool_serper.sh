#!/usr/bin/env bash
set -euo pipefail

# 当前脚本所在目录：.../evaluation/src/tools
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# 项目根目录：回到 .../efficiency_rl
ROOT_DIR="$( cd "$SCRIPT_DIR/../../.." &> /dev/null && pwd )"
cd "$ROOT_DIR"
echo "Switched to project root: $ROOT_DIR"

# 确保以项目根为 PYTHONPATH，供模块化运行
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# 以模块方式运行，保证包内相对导入生效
python -u -m evaluation.src.tools.search_tool_serper "$@"
