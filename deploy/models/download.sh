#!/usr/bin/env bash
# Afterglow — 编译 llama.cpp(CUDA) 并下载 GGUF 模型。幂等，可重复执行。
#
# 用法:
#   deploy/models/download.sh [--with-brain] [--skip-build] [--skip-download]
#
# 环境变量(均可覆盖):
#   AFTERGLOW_DATA   数据根目录            默认 /data/afterglow
#   GGUF_DIR         GGUF 下载目录         默认 $AFTERGLOW_DATA/models/gguf
#   LLAMA_DIR        llama.cpp 源码/编译   默认 /opt/llama.cpp
#   CUDA_ARCH        例如 "89"(RTX 2000 Ada)；留空 = 让 CMake 自动检测
#   BUILD_JOBS       并行编译数            默认 min(nproc, 8)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK="${SCRIPT_DIR}/models.lock"
AFTERGLOW_DATA="${AFTERGLOW_DATA:-/data/afterglow}"
GGUF_DIR="${GGUF_DIR:-${AFTERGLOW_DATA}/models/gguf}"
LLAMA_DIR="${LLAMA_DIR:-/opt/llama.cpp}"
CUDA_ARCH="${CUDA_ARCH:-}"
BUILD_JOBS="${BUILD_JOBS:-$(( $(nproc) < 8 ? $(nproc) : 8 ))}"

WITH_BRAIN=0
SKIP_BUILD=0
SKIP_DOWNLOAD=0

log()  { printf '\033[1;34m[download]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[download] 警告:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[download] 错误:\033[0m %b\n' "$*" >&2; exit 1; }
step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-brain)    WITH_BRAIN=1 ;;
    --skip-build)    SKIP_BUILD=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    -h|--help)       usage; exit 0 ;;
    *) die "未知参数: $1(见 --help)" ;;
  esac
  shift
done

[[ -f "$LOCK" ]] || die "找不到 $LOCK"
command -v python3 >/dev/null || die "需要 python3(用于解析 models.lock)"

# ---- models.lock 解析(TOML, python3 标准库 tomllib; -I = 隔离模式) ----
lock_value() { # section key
  python3 -I - "$LOCK" "$1" "$2" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    d = tomllib.load(f)
print(d[sys.argv[2]][sys.argv[3]])
PY
}

lock_models() { # 输出 TSV: name role required file url sha256
  python3 -I - "$LOCK" <<'PY'
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    d = tomllib.load(f)
for m in d.get("model", []):
    print("\t".join([m["name"], m["role"], "1" if m.get("required") else "0", m["file"], m["url"], m["sha256"]]))
PY
}

lock_set_sha() { # file sha256 —— 只改对应 [[model]] 块里的 sha256 行，保留注释
  python3 -I - "$LOCK" "$1" "$2" <<'PY'
import re, sys
path, fname, sha = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
parts = re.split(r"(?m)^(?=\[\[model\]\])", text)
done = False
for i, block in enumerate(parts):
    if block.startswith("[[model]]") and re.search(r'(?m)^file\s*=\s*"%s"' % re.escape(fname), block):
        parts[i] = re.sub(r'(?m)^(sha256\s*=\s*)"[^"]*"', r'\g<1>"%s"' % sha, block, count=1)
        done = True
if not done:
    sys.exit("models.lock: 没有找到文件 %s" % fname)
tmp = path + ".tmp"
open(tmp, "w", encoding="utf-8").write("".join(parts))
import os
os.replace(tmp, path)
PY
}

ensure_dir() { # 需要时用 sudo 创建并交给当前用户
  local d="$1"
  if [[ ! -d "$d" ]]; then
    mkdir -p "$d" 2>/dev/null || { sudo mkdir -p "$d"; }
  fi
  if [[ ! -w "$d" ]]; then
    sudo chown -R "$(id -u):$(id -g)" "$d"
  fi
}

# ---------------------------------------------------------------- 1. llama.cpp
build_llama() {
  step "1/2 编译 llama.cpp (CUDA) → ${LLAMA_DIR}"
  local repo tag
  repo="$(lock_value llama_cpp repo)"
  tag="$(lock_value llama_cpp tag)"

  command -v cmake >/dev/null || die "缺少 cmake(install.sh 第 1 步会安装)"
  command -v git   >/dev/null || die "缺少 git"
  if ! command -v nvcc >/dev/null && [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
    warn "没有找到 nvcc(CUDA Toolkit)。尝试 apt 安装 nvidia-cuda-toolkit(Ubuntu 仓库版本)..."
    sudo apt-get install -y nvidia-cuda-toolkit
  fi
  if ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
  fi

  ensure_dir "$LLAMA_DIR"
  if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
    log "git clone ${repo}"
    git clone "$repo" "$LLAMA_DIR"
  fi
  git -C "$LLAMA_DIR" fetch --tags --quiet origin || warn "git fetch 失败，使用本地已有对象"

  local ref
  if git -C "$LLAMA_DIR" rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    git -C "$LLAMA_DIR" checkout --quiet "tags/${tag}"
    ref="$tag"
  else
    warn "标签 ${tag} 不存在，回退到默认分支最新提交(可能不稳定)"
    local branch
    branch="$(git -C "$LLAMA_DIR" symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##' || true)"
    branch="${branch:-master}"
    git -C "$LLAMA_DIR" checkout --quiet "$branch"
    git -C "$LLAMA_DIR" merge --ff-only --quiet "origin/${branch}" || warn "快进合并失败，保持当前提交"
    ref="${branch}@$(git -C "$LLAMA_DIR" rev-parse --short HEAD)"
  fi

  local head marker
  head="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
  marker="${LLAMA_DIR}/build/.afterglow-built"
  if [[ -x "${LLAMA_DIR}/build/bin/llama-server" && -f "$marker" && "$(cat "$marker")" == "$head" ]]; then
    log "llama.cpp ${ref} 已编译，跳过"
    return 0
  fi

  # CUDA 构建: https://huggingface.co/hekod19045/llama-cuda/blob/main/docs/build.md (docs/build.md 副本)
  local -a cmake_args=(-B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release)
  [[ -n "$CUDA_ARCH" ]] && cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
  log "cmake ${cmake_args[*]}"
  (cd "$LLAMA_DIR" && cmake "${cmake_args[@]}")
  log "cmake --build (jobs=${BUILD_JOBS})，CUDA 编译需要几分钟到二十几分钟"
  (cd "$LLAMA_DIR" && cmake --build build --config Release -j "$BUILD_JOBS")

  if [[ ! -x "${LLAMA_DIR}/build/bin/llama-server" ]]; then
    warn "没有生成 build/bin/llama-server。新版 llama.cpp 可能改了二进制名(模型卡提到 'llama serve' / llama-app 目标)。"
    warn "请查看 ${LLAMA_DIR}/build/bin/ 并相应修改 deploy/systemd/afterglow-llm.service 的 ExecStart。"
    return 1
  fi
  echo "$head" > "$marker"
  log "编译完成: ${LLAMA_DIR}/build/bin/llama-server ($ref)"
}

# ---------------------------------------------------------------- 2. GGUF
sha_of() { sha256sum "$1" | awk '{print $1}'; }

fetch_model() { # name role required file url sha256
  local name="$1" role="$2" file="$4" url="$5" want="$6"
  local dest="${GGUF_DIR}/${file}"

  if [[ ! -f "$dest" ]]; then
    log "下载 ${name}/${role}: ${file}"
    log "来源: ${url}"
    # -C - = 断点续传；先写 .part，完成后再改名，避免半截文件被当成成品
    curl -L -C - --fail --retry 5 --retry-delay 5 --connect-timeout 30 \
         -o "${dest}.part" "$url"
    mv "${dest}.part" "$dest"
  else
    log "已存在: ${file}"
  fi

  local got
  got="$(sha_of "$dest")"
  if [[ "$want" =~ ^[0-9a-f]{64}$ ]]; then
    if [[ "$got" != "$want" ]]; then
      mv "$dest" "${dest}.bad"
      die "${file} sha256 不匹配\n  期望 ${want}\n  实际 ${got}\n  已改名为 ${file}.bad，请删除后重新运行。"
    fi
    log "sha256 校验通过: ${file}"
  else
    log "首次下载，记录 sha256: ${file}"
    printf '    sha256 = %s\n' "$got"
    lock_set_sha "$file" "$got"
    warn "已写入 models.lock。请与 Hugging Face 文件页的 LFS SHA256 对照确认。"
  fi
}

download_models() {
  step "2/2 下载 GGUF 模型 → ${GGUF_DIR}"
  command -v curl >/dev/null || die "缺少 curl"
  ensure_dir "$GGUF_DIR"
  local avail_gb
  avail_gb="$(df -BG --output=avail "$GGUF_DIR" | tail -1 | tr -dc '0-9')"
  local need=4; [[ $WITH_BRAIN -eq 1 ]] && need=30
  if [[ -n "$avail_gb" && "$avail_gb" -lt "$need" ]]; then
    die "${GGUF_DIR} 可用空间 ${avail_gb} GB，至少需要约 ${need} GB"
  fi

  local line name role required file url sha
  while IFS=$'\t' read -r name role required file url sha; do
    if [[ "$required" != "1" && $WITH_BRAIN -ne 1 ]]; then
      log "跳过可选模型 ${name}/${role}(加 --with-brain 下载)"
      continue
    fi
    fetch_model "$name" "$role" "$required" "$file" "$url" "$sha"
    # lock_set_sha 改写了文件，后续行里的 sha 仍是旧值，但各行文件不同，不影响
  done < <(lock_models)
}

[[ $SKIP_BUILD -eq 1 ]] || build_llama
[[ $SKIP_DOWNLOAD -eq 1 ]] || download_models
step "完成"
log "llama-server: ${LLAMA_DIR}/build/bin/llama-server"
log "GGUF 目录:    ${GGUF_DIR}"
