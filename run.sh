#!/usr/bin/env bash
set -euo pipefail

# ==== 配置 ====
ENV_NAME="asr"
PY_VER="3.9"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# ==== 检查并安装 ffmpeg ====
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "⚙️ 安装 ffmpeg ..."
  yum install -y epel-release
  rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
  rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm || true
  yum install -y ffmpeg ffmpeg-devel
else
  echo "✅ 已检测到 ffmpeg"
fi

# ==== Conda 环境 ====
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ 未检测到 conda，请先安装 Miniconda/Anaconda 后重试。"
  exit 1
fi

# 创建环境（如果不存在）
if ! conda env list | grep -q "^${ENV_NAME}\s"; then
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# 安装依赖
pip install -r "$APP_DIR/requirements.txt"

# 检查 Vosk 模型
MODEL_DIR="$APP_DIR/models/vosk-model-small-cn-0.22"
if [ ! -d "$MODEL_DIR" ]; then
  echo "⬇️  请下载并解压 Vosk 小中文模型到：$MODEL_DIR"
  echo "   搜索：vosk-model-small-cn-0.22  下载 zip 后解压到 models/ 目录。"
  exit 1
fi

# ==== 启动 Flask ====
export FLASK_ENV=production
python "$APP_DIR/app.py"
