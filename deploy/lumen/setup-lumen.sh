#!/usr/bin/env bash
# Afterglow — 安装 LUMEN RAW(源码方式，MIT)。幂等。
#
# 做的事（与 lumen-raw 自带的 run-source.sh 保持一致，只是不启动 GUI）:
#   1. git clone/pin lumen-raw 到 $LUMEN_DIR(默认 /opt/lumen-raw)，提交号取自 deploy/models/models.lock
#   2. python3.12 -m venv .venv-linux
#   3. pip install -r requirements-linux.txt(含 requirements.txt)
#      有 NVIDIA GPU 时：卸载 CPU 版 onnxruntime，再 pip install -r requirements-linux-gpu.txt
#      (requirements-linux-gpu.txt 文件头注释要求这个顺序)
#   4. tools/fetch_assets.py + tools/fetch_exiftool.py(run-source.sh 同样会做；运行时素材、ExifTool)
#   5. main.py --cli models --download   (LUMEN_MODELS=$LUMEN_MODELS_DIR)
#   6. main.py --cli commands            (冒烟测试)
#
# 环境变量:
#   LUMEN_DIR         默认 /opt/lumen-raw
#   LUMEN_COMMIT      默认读 models.lock 的 [lumen].commit
#   LUMEN_MODELS_DIR  默认 /data/afterglow/models/lumen   (传给 LUMEN_MODELS)
#   LUMEN_LIBRARY     默认 /data/afterglow/lumen-library.db
#   LUMEN_PYTHON      基础解释器，默认 python3.12
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${SCRIPT_DIR}/../models/models.lock"
AFTERGLOW_DATA="${AFTERGLOW_DATA:-/data/afterglow}"
LUMEN_DIR="${LUMEN_DIR:-/opt/lumen-raw}"
LUMEN_MODELS_DIR="${LUMEN_MODELS_DIR:-${AFTERGLOW_DATA}/models/lumen}"
LUMEN_LIBRARY="${LUMEN_LIBRARY:-${AFTERGLOW_DATA}/lumen-library.db}"
PYTHON_BASE="${LUMEN_PYTHON:-python3.12}"

log()  { printf '\033[1;34m[lumen]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[lumen] 警告:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[lumen] 错误:\033[0m %s\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

lock_value() {
  python3 -I - "$LOCK" "$1" "$2" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    d = tomllib.load(f)
print(d[sys.argv[2]][sys.argv[3]])
PY
}

ensure_dir() {
  local d="$1"
  [[ -d "$d" ]] || mkdir -p "$d" 2>/dev/null || sudo mkdir -p "$d"
  [[ -w "$d" ]] || sudo chown -R "$(id -u):$(id -g)" "$d"
}

command -v git >/dev/null || die "缺少 git"
command -v "$PYTHON_BASE" >/dev/null || die "缺少 ${PYTHON_BASE}(Ubuntu 24.04: sudo apt install python3.12 python3.12-venv)"
command -v perl >/dev/null || warn "没有 perl；LUMEN 读取 EXIF/镜头信息需要它(sudo apt install perl)"

REPO="$(lock_value lumen repo)"
COMMIT="${LUMEN_COMMIT:-$(lock_value lumen commit)}"

# ---------------------------------------------------------------- 1. clone + pin
step "1/6 获取 lumen-raw ${COMMIT:0:12} → ${LUMEN_DIR}"
ensure_dir "$LUMEN_DIR"
if [[ ! -d "${LUMEN_DIR}/.git" ]]; then
  git clone "$REPO" "$LUMEN_DIR"
fi
if [[ "$(git -C "$LUMEN_DIR" rev-parse HEAD)" != "$COMMIT" ]]; then
  git -C "$LUMEN_DIR" fetch --quiet origin || warn "git fetch 失败，尝试使用本地对象"
  git -C "$LUMEN_DIR" checkout --quiet "$COMMIT" || die "无法检出 ${COMMIT}"
fi
log "HEAD = $(git -C "$LUMEN_DIR" rev-parse HEAD)"

cd "$LUMEN_DIR"

# ---------------------------------------------------------------- 2. venv
step "2/6 创建 .venv-linux (${PYTHON_BASE})"
if [[ ! -x .venv-linux/bin/python ]]; then
  "$PYTHON_BASE" -m venv .venv-linux
fi
PY="${LUMEN_DIR}/.venv-linux/bin/python"
"$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
pip() { "$PY" -m pip --disable-pip-version-check "$@"; }

# CUDA/cuDNN 的 pip 轮子装在 site-packages/nvidia 下，需要在加载前放进 LD_LIBRARY_PATH
# (run-source.sh 同样调用 lumen.cuda_env.prepare)。
cuda_env() {
  local libs
  libs="$("$PY" -c 'from lumen.cuda_env import prepare; print(prepare())' 2>/dev/null || true)"
  if [[ -n "$libs" ]]; then
    export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

# ---------------------------------------------------------------- 3. requirements
step "3/6 安装 Python 依赖"
STAMP=".venv-linux/afterglow-requirements.stamp"
cat requirements.txt requirements-linux.txt requirements-linux-gpu.txt > "${STAMP}.new"
HAVE_GPU=0
command -v nvidia-smi >/dev/null 2>&1 && HAVE_GPU=1

gpu_ok() { "$PY" -c "import cupy, onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()" >/dev/null 2>&1; }

need_install=0
cmp -s "${STAMP}.new" "$STAMP" 2>/dev/null || need_install=1
cuda_env
if [[ $HAVE_GPU -eq 1 ]] && ! gpu_ok; then need_install=1; fi
"$PY" -c "import numpy, cv2, PIL, rawpy, PySide6.QtMultimedia, tifffile, onnxruntime, av" >/dev/null 2>&1 || need_install=1

if [[ $need_install -eq 1 ]]; then
  log "pip install -r requirements-linux.txt"
  pip install -r requirements-linux.txt
  if [[ $HAVE_GPU -eq 1 ]]; then
    log "检测到 nvidia-smi：换装 GPU 版 onnxruntime(requirements-linux-gpu.txt)"
    # 不能让 CPU 版和 GPU 版 onnxruntime 并存(它们占用同一批模块)
    pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
    pip install -r requirements-linux-gpu.txt
  else
    warn "没有 nvidia-smi：只装 CPU 版，LUMEN 将用 CPU 运行(很慢)"
  fi
  mv "${STAMP}.new" "$STAMP"
else
  log "依赖已是最新，跳过"
  rm -f "${STAMP}.new"
fi
cuda_env

# ---------------------------------------------------------------- 4. assets
step "4/6 运行时素材与 ExifTool (tools/fetch_assets.py, tools/fetch_exiftool.py)"
"$PY" tools/fetch_assets.py --verify-only >/dev/null 2>&1 || "$PY" tools/fetch_assets.py
"$PY" tools/fetch_exiftool.py

# ---------------------------------------------------------------- 5. AI models
step "5/6 下载 LUMEN AI 模型 → ${LUMEN_MODELS_DIR}"
ensure_dir "$LUMEN_MODELS_DIR"
ensure_dir "$(dirname "$LUMEN_LIBRARY")"
export LUMEN_MODELS="$LUMEN_MODELS_DIR"
export LUMEN_LIBRARY
export ORT_DISABLE_TELEMETRY=1
# 退出码: 0 = 全部就绪, 2 = 仍有缺失, 1 = 下载出错 (lumen/cli.py: models())
"$PY" main.py --cli models --download

# ---------------------------------------------------------------- 6. smoke test
step "6/6 冒烟测试: main.py --cli commands"
if "$PY" main.py --cli commands >/tmp/afterglow-lumen-commands.json; then
  log "OK，可用命令 $(wc -c </tmp/afterglow-lumen-commands.json) 字节 → /tmp/afterglow-lumen-commands.json"
else
  die "main.py --cli commands 失败"
fi

step "LUMEN RAW 就绪"
log "python:  ${PY}"
log "models:  ${LUMEN_MODELS_DIR}"
log "library: ${LUMEN_LIBRARY}"
