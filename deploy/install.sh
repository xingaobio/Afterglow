#!/usr/bin/env bash
# Afterglow — Ubuntu 24.04 一键安装。幂等，可重复执行。以普通用户运行，需要 sudo 时会提示输入密码。
#
# 用法: deploy/install.sh [--with-brain] [--skip-immich] [--enable-watch] [-h]
#   --with-brain    同时下载 Qwen3.6-35B-A3B(约 23 GB)并安装 afterglow-brain.service
#   --skip-immich   不装 Docker / NVIDIA Container Toolkit / Immich
#   --enable-watch  enable afterglow-watch.service
#
# 环境变量: AFTERGLOW_DATA(/data/afterglow) IMMICH_DIR(/opt/immich) AFTERGLOW_OPT(/opt/afterglow)
#           PHOTO_SOURCE(默认读 ~/.config/afterglow/afterglow.toml 的 paths.source，否则 /media/photos)
#
# 退出码: 0 成功, 1 一般错误, 3 外部依赖不可用(nvidia-smi 缺失)
#
# 外部步骤的官方来源:
#   Docker Engine (apt):  https://docs.docker.com/engine/install/ubuntu/
#   NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#   Immich compose:       https://docs.immich.app/install/docker-compose
#   Immich CUDA ML:       https://docs.immich.app/features/ml-hardware-acceleration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AFTERGLOW_DATA="${AFTERGLOW_DATA:-/data/afterglow}"
AFTERGLOW_OPT="${AFTERGLOW_OPT:-/opt/afterglow}"
IMMICH_DIR="${IMMICH_DIR:-/opt/immich}"
RUN_USER="$(id -un)"
CONFIG_FILE="${HOME}/.config/afterglow/afterglow.toml"

WITH_BRAIN=0
SKIP_IMMICH=0
ENABLE_WATCH=0

log()  { printf '\033[1;34m[afterglow]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[afterglow] 警告:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[afterglow] 错误:\033[0m %b\n' "$*" >&2; exit "${2:-1}"; }
step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

usage() { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-brain)   WITH_BRAIN=1 ;;
    --skip-immich)  SKIP_IMMICH=1 ;;
    --enable-watch) ENABLE_WATCH=1 ;;
    -h|--help)      usage; exit 0 ;;
    *) die "未知参数: $1(见 --help)" ;;
  esac
  shift
done

# ---------------------------------------------------------------- 0. 前置检查
step "0/9 前置检查"
[[ $EUID -ne 0 ]] || die "请用普通用户运行(脚本会在需要时调用 sudo)，不要直接用 root。"
command -v sudo >/dev/null || die "需要 sudo"
command -v apt-get >/dev/null || die "只支持 Debian/Ubuntu 系(需要 apt-get)；目标平台是 Ubuntu 24.04。"
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    warn "检测到 ${PRETTY_NAME:-未知系统}；Afterglow 的目标平台是 Ubuntu 24.04 LTS，其它系统未测试。"
  fi
fi
sudo -v
free_gb="$(df -BG --output=avail "$(dirname "$AFTERGLOW_DATA")" 2>/dev/null | tail -1 | tr -dc '0-9' || true)"
need_gb=40; [[ $WITH_BRAIN -eq 1 ]] && need_gb=70
if [[ -n "${free_gb:-}" && "$free_gb" -lt "$need_gb" ]]; then
  warn "$(dirname "$AFTERGLOW_DATA") 可用空间约 ${free_gb} GB；建议至少 ${need_gb} GB(不含预览/输出，1 TB 素材的工作盘另需数百 GB)。"
fi

ensure_dir() { # 创建目录并交给当前用户
  local d="$1"
  [[ -d "$d" ]] || mkdir -p "$d" 2>/dev/null || sudo mkdir -p "$d"
  [[ -w "$d" ]] || sudo chown -R "$(id -u):$(id -g)" "$d"
}

# ---------------------------------------------------------------- 1. apt
step "1/9 安装系统包 (apt)"
APT_PKGS=(exiftool libimage-exiftool-perl rsync ffmpeg git build-essential cmake python3.12 python3.12-venv
          python3-venv curl jq ca-certificates gnupg perl pkg-config libcurl4-openssl-dev)
sudo apt-get update
# exiftool 在部分 Ubuntu 版本里只是 libimage-exiftool-perl 的别名；逐个装，缺哪个跳过哪个，最后再校验命令。
for p in "${APT_PKGS[@]}"; do
  if dpkg -s "$p" >/dev/null 2>&1; then continue; fi
  sudo apt-get install -y "$p" || warn "apt 包 $p 安装失败/不存在，稍后按命令是否可用再判断"
done
for c in exiftool rsync ffmpeg git cmake python3.12 curl jq perl; do
  command -v "$c" >/dev/null || die "安装后仍找不到命令: $c"
done
log "系统包就绪"

# ---------------------------------------------------------------- 2. NVIDIA 驱动
step "2/9 检查 NVIDIA 驱动 (nvidia-smi)"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[afterglow] 错误: nvidia-smi 不可用 —— 没有可用的 NVIDIA 驱动。
  Afterglow 的视觉模型、LUMEN、Immich ML 都需要 GPU。请先安装驱动并重启:
      sudo ubuntu-drivers install
      sudo reboot
  然后重新运行本脚本。
MSG
  exit 3
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/[afterglow] GPU: /'

# ---------------------------------------------------------------- 3. Docker + NVIDIA Container Toolkit
step "3/9 Docker Engine + compose 插件 + NVIDIA Container Toolkit"
if [[ $SKIP_IMMICH -eq 1 ]]; then
  log "--skip-immich：跳过"
else
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker 与 compose 插件已安装，跳过"
  else
    # 官方 apt 源安装 (不是 Docker Desktop)。来源: https://docs.docker.com/engine/install/ubuntu/
    if dpkg -l docker.io docker-compose podman-docker 2>/dev/null | grep -q '^ii'; then
      warn "检测到非官方的 docker.io/docker-compose/podman-docker 包。官方文档要求先卸载它们: sudo apt remove docker.io docker-compose podman-docker"
      die "请手动卸载后重跑。"
    fi
    sudo apt-get install -y ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<DOCKERSRC
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
DOCKERSRC
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi
  sudo systemctl enable --now docker
  if ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx docker; then
    sudo usermod -aG docker "$RUN_USER"
    warn "已把 ${RUN_USER} 加入 docker 组，重新登录后生效；本次脚本改用 sudo docker。"
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "NVIDIA Container Toolkit 已安装，跳过"
  else
    # 来源: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    # (官方示例用 NVIDIA_CONTAINER_TOOLKIT_VERSION 固定版本；这里不固定，装仓库里的最新版。)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes \
      -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  fi
  before="$(sudo sha256sum /etc/docker/daemon.json 2>/dev/null || true)"
  sudo nvidia-ctk runtime configure --runtime=docker
  after="$(sudo sha256sum /etc/docker/daemon.json 2>/dev/null || true)"
  if [[ "$before" != "$after" ]]; then
    sudo systemctl restart docker
    log "Docker 运行时配置已更新并重启"
  fi
fi

# ---------------------------------------------------------------- 4. LUMEN
step "4/9 LUMEN RAW (deploy/lumen/setup-lumen.sh)"
AFTERGLOW_DATA="$AFTERGLOW_DATA" bash "${SCRIPT_DIR}/lumen/setup-lumen.sh"

# ---------------------------------------------------------------- 5. llama.cpp + 模型
step "5/9 llama.cpp (CUDA) 与 GGUF 模型 (deploy/models/download.sh)"
dl_args=()
[[ $WITH_BRAIN -eq 1 ]] && dl_args+=(--with-brain)
AFTERGLOW_DATA="$AFTERGLOW_DATA" bash "${SCRIPT_DIR}/models/download.sh" "${dl_args[@]}"

# ---------------------------------------------------------------- 6. Afterglow 本体
step "6/9 Afterglow 本体 → ${AFTERGLOW_OPT}/.venv"
ensure_dir "$AFTERGLOW_OPT"
if [[ ! -x "${AFTERGLOW_OPT}/.venv/bin/python" ]]; then
  python3.12 -m venv "${AFTERGLOW_OPT}/.venv"
fi
"${AFTERGLOW_OPT}/.venv/bin/python" -m pip install --disable-pip-version-check --upgrade pip
# 可编辑安装：改仓库代码立即生效。仓库不要删/移动。
"${AFTERGLOW_OPT}/.venv/bin/python" -m pip install --disable-pip-version-check -e "${REPO_DIR}"
ensure_dir "$AFTERGLOW_DATA"
if [[ ! -f "$CONFIG_FILE" ]]; then
  mkdir -p "$(dirname "$CONFIG_FILE")"
  if [[ -f "${REPO_DIR}/config/afterglow.example.toml" ]]; then
    cp "${REPO_DIR}/config/afterglow.example.toml" "$CONFIG_FILE"
    log "已创建配置: ${CONFIG_FILE}(请编辑 paths.source 等)"
  else
    warn "找不到 config/afterglow.example.toml，未创建配置文件"
  fi
fi
mkdir -p "${HOME}/.local/bin"
ln -sf "${AFTERGLOW_OPT}/.venv/bin/afterglow" "${HOME}/.local/bin/afterglow"

# ---------------------------------------------------------------- 7. Immich
step "7/9 Immich (docker compose) → ${IMMICH_DIR}"
if [[ $SKIP_IMMICH -eq 1 ]]; then
  log "--skip-immich：跳过"
else
  ensure_dir "$IMMICH_DIR"
  ensure_dir "$IMMICH_DIR/official"
  cp "${SCRIPT_DIR}/immich/docker-compose.yml" "${IMMICH_DIR}/docker-compose.yml"

  # 下载当前官方文件，供对照；并从中提取 database / valkey 镜像标签
  # 官方下载地址来自 https://docs.immich.app/install/docker-compose
  for f in docker-compose.yml example.env hwaccel.ml.yml; do
    curl -fsSL "https://github.com/immich-app/immich/releases/latest/download/${f}" -o "${IMMICH_DIR}/official/${f}" \
      || warn "无法下载官方 ${f}(离线？)"
  done

  # PHOTO_SOURCE: 环境变量 > 配置文件 > 默认
  if [[ -z "${PHOTO_SOURCE:-}" && -f "$CONFIG_FILE" ]]; then
    PHOTO_SOURCE="$(python3 -I - "$CONFIG_FILE" <<'PY' || true
import sys, tomllib
try:
    with open(sys.argv[1], "rb") as f:
        print(tomllib.load(f).get("paths", {}).get("source", ""))
except Exception:
    pass
PY
)"
  fi
  PHOTO_SOURCE="${PHOTO_SOURCE:-/media/photos}"

  official_image() { # 服务名 → 官方 compose 里该服务的 image: 行
    awk -v svc="$1" '
      /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { cur=$1; sub(/:$/, "", cur); next }
      cur==svc && $1=="image:" { sub(/^[[:space:]]*image:[[:space:]]*/, ""); print; exit }
    ' "${IMMICH_DIR}/official/docker-compose.yml" 2>/dev/null || true
  }

  if [[ ! -f "${IMMICH_DIR}/.env" ]]; then
    cp "${SCRIPT_DIR}/immich/example.env" "${IMMICH_DIR}/.env"
    db_pw="$(head -c 64 /dev/urandom | tr -dc 'A-Za-z0-9' | head -c 24)"
    sed -i \
      -e "s|^DB_PASSWORD=.*|DB_PASSWORD=${db_pw}|" \
      -e "s|^UPLOAD_LOCATION=.*|UPLOAD_LOCATION=${AFTERGLOW_DATA}/immich/upload|" \
      -e "s|^DB_DATA_LOCATION=.*|DB_DATA_LOCATION=${AFTERGLOW_DATA}/immich/postgres|" \
      -e "s|^AFTERGLOW_ORGANIZED=.*|AFTERGLOW_ORGANIZED=${AFTERGLOW_DATA}/library|" \
      -e "s|^PHOTO_SOURCE=.*|PHOTO_SOURCE=${PHOTO_SOURCE}|" \
      "${IMMICH_DIR}/.env"
    chmod 600 "${IMMICH_DIR}/.env"
    log "已生成 ${IMMICH_DIR}/.env(随机数据库密码)"
  else
    log "${IMMICH_DIR}/.env 已存在，保留(不覆盖密码)"
  fi

  # 填 database / valkey 镜像（仅在 .env 里为空时）
  for pair in "IMMICH_DB_IMAGE:database" "IMMICH_VALKEY_IMAGE:redis"; do
    var="${pair%%:*}"; svc="${pair##*:}"
    if ! grep -Eq "^${var}=.+" "${IMMICH_DIR}/.env"; then
      img="$(official_image "$svc")"
      [[ -z "$img" && "$svc" == "redis" ]] && img="$(official_image valkey)"
      if [[ -n "$img" ]]; then
        sed -i "s|^${var}=.*|${var}=${img}|" "${IMMICH_DIR}/.env"
        log "${var}=${img}(取自官方 compose)"
      else
        warn "没能从官方 compose 提取 ${var}；请手动编辑 ${IMMICH_DIR}/.env(见 ${IMMICH_DIR}/official/docker-compose.yml 的 ${svc} 服务)"
      fi
    fi
  done

  mkdir -p "${AFTERGLOW_DATA}/immich/upload" "${AFTERGLOW_DATA}/immich/postgres" "${AFTERGLOW_DATA}/library"

  DOCKER=(docker)
  docker info >/dev/null 2>&1 || DOCKER=(sudo docker)
  if grep -Eq '^IMMICH_(DB|VALKEY)_IMAGE=$' "${IMMICH_DIR}/.env"; then
    warn "IMMICH_DB_IMAGE / IMMICH_VALKEY_IMAGE 未填，暂不启动 Immich。填好后: cd ${IMMICH_DIR} && docker compose up -d"
  else
    (cd "$IMMICH_DIR" && "${DOCKER[@]}" compose up -d)
    log "Immich 已启动: http://127.0.0.1:2283 (首次访问创建管理员，然后在 账户设置 → API 密钥 创建 key)"
  fi
fi

# ---------------------------------------------------------------- 8. systemd + udev
step "8/9 systemd 单元与 udev 规则"
units=(afterglow-llm.service afterglow-watch.service afterglow-nightly.service afterglow-nightly.timer afterglow-run@.service)
[[ $WITH_BRAIN -eq 1 ]] && units+=(afterglow-brain.service)
for u in "${units[@]}"; do
  sed "s/@AFTERGLOW_USER@/${RUN_USER}/g" "${SCRIPT_DIR}/systemd/${u}" | sudo tee "/etc/systemd/system/${u}" >/dev/null
done
sudo cp "${SCRIPT_DIR}/systemd/99-afterglow-usb.rules" /etc/udev/rules.d/99-afterglow-usb.rules
sudo systemctl daemon-reload
sudo udevadm control --reload
if [[ -f /data/afterglow/models/gguf/Qwen3VL-4B-Instruct-Q4_K_M.gguf || -f "${AFTERGLOW_DATA}/models/gguf/Qwen3VL-4B-Instruct-Q4_K_M.gguf" ]]; then
  sudo systemctl enable --now afterglow-llm.service
else
  warn "没有找到视觉模型 GGUF，未启动 afterglow-llm.service"
fi
sudo systemctl enable --now afterglow-nightly.timer
[[ $ENABLE_WATCH -eq 1 ]] && sudo systemctl enable --now afterglow-watch.service
log "已安装: ${units[*]} 99-afterglow-usb.rules(默认规则对任何带卷标的 USB 盘生效，见 deploy/systemd/README.md)"

# ---------------------------------------------------------------- 9. 下一步
step "9/9 完成 —— 下一步"
cat <<MSG
  1. 编辑配置:   ${CONFIG_FILE}
                 (paths.source / paths.backup / paths.organized；[lumen]、[llm] 默认值已匹配本脚本的安装位置)
  2. Immich:     打开 http://127.0.0.1:2283 创建管理员，在 账户设置 → API 密钥 新建 key，填入配置 [immich].api_key
  3. 检查:       afterglow doctor
  4. 初始化:     afterglow setup
  5. 试运行:     afterglow run --dry-run
  6. 正式运行:   afterglow run

  若刚被加入 docker 组：重新登录一次，之后可直接用 docker compose。
  PATH: 确保 ~/.local/bin 在 PATH 中(afterglow 的软链接在那里)。
MSG
