#!/usr/bin/env bash
# One-time setup on the EC2 GPU instance (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 22.04/24.04).
# Usage: bash ~/business_entity_resolution/scripts/aws_setup.sh ~/6ab10eb3b23ba_student_resource.zip
set -euo pipefail
ZIP=${1:?path to 6ab10eb3b23ba_student_resource.zip}
REPO=$(cd "$(dirname "$0")/.." && pwd)
sudo apt-get update -qq && sudo apt-get install -y -qq unzip tmux htop >/dev/null || true
# Fast local NVMe instance store (g6e/g5/p4d have one) -> /data ; otherwise use the EBS root volume.
# NOTE: instance-store data is LOST when the instance is stopped. Copy outputs off the box before stopping.
sudo mkdir -p /data
if ! mountpoint -q /data; then
  DEV=$(lsblk -dpno NAME,MODEL | grep -i "Instance Storage" | head -1 | awk '{print $1}' || true)
  if [ -n "${DEV:-}" ]; then
    echo "formatting instance store $DEV -> /data"
    sudo mkfs.ext4 -q -F "$DEV" && sudo mount "$DEV" /data
  fi
fi
sudo chown "$USER" /data
cd /data
[ -d student_resource ] || unzip -q "$ZIP" -x "__MACOSX/*"
ls /data/student_resource/dataset/train /data/student_resource/dataset/test
# Python 3.12 venv via uv (the pinned pandas/numpy need Python >= 3.11; Ubuntu 22.04 ships 3.10)
python3 -m pip install --user -q uv || pip install --user -q uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /data/venv
source /data/venv/bin/activate
uv pip install -q torch                                   # PyPI Linux wheel bundles the CUDA runtime
grep -v -E "^(torch|faiss-cpu)" "$REPO/requirements.txt" | grep -v "^#" | grep -v "^$" > /tmp/req.txt
uv pip install -q -r /tmp/req.txt
uv pip install -q faiss-gpu-cu12 || uv pip install -q faiss-cpu==1.15.1
python - <<'EOF'
import torch, faiss, lightgbm, xgboost, catboost, rapidfuzz, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("faiss gpus", faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0, "| pandas", pandas.__version__)
EOF
cd "$REPO" && PYTHONPATH=src python -m pytest -q tests && echo "SETUP OK -> next: tmux new -s ber ; bash scripts/run_all.sh configs/aws.yaml gbdt"
