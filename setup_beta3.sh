#!/usr/bin/env bash
# Beta 3 PyTorch Native install per OpencodeDocs/steering/pytorch-native.md
# The SDK 2.31 DLAMI already has docker-ce, aws-cli, python3.12 -- skip those steps.

set -euo pipefail

echo "===== Verify pre-installed tools ====="
docker --version
aws --version
python3.12 --version
echo

echo "===== Step 3: ECR login and pull Beta 3 image ====="
# ECR password was pushed to /tmp/ecr_pw from the operator machine (instance has no IAM role).
if [ ! -s /tmp/ecr_pw ]; then
  echo "FATAL: /tmp/ecr_pw missing or empty. Push a fresh ECR token from operator machine."
  exit 1
fi
cat /tmp/ecr_pw | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
sudo docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
IMAGE_ID=$(sudo docker images -q --filter reference=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest)
echo "IMAGE_ID=$IMAGE_ID"

echo "===== Step 4: Extract /workspace from container ====="
cd $HOME
sudo docker rm -f tmp_beta3 2>/dev/null || true
sudo docker create --name tmp_beta3 $IMAGE_ID
sudo docker cp tmp_beta3:/workspace .
sudo docker rm tmp_beta3
sudo chown -R $USER:$USER $HOME/workspace
ls $HOME/workspace/

echo "===== Step 5: Install Beta 3 host runtime (CRITICAL) ====="
sudo apt-get install -y dkms build-essential python3.12-venv
ls $HOME/workspace/runtime_artifacts/
# Bypass any held packages / force reinstall over DLAMI versions
sudo dpkg -i --force-confnew --force-overwrite $HOME/workspace/runtime_artifacts/*.deb || {
  echo "First dpkg pass failed, trying --configure -a then again"
  sudo dpkg --configure -a
  sudo dpkg -i --force-confnew --force-overwrite $HOME/workspace/runtime_artifacts/*.deb
}
dpkg -l | grep aws-neuronx | head -10

echo "===== Step 6: Create Beta 3 venv ====="
cd $HOME/workspace
python3.12 -m venv native_venv
source native_venv/bin/activate
python --version

pip install --upgrade pip
pip install uv
export UV_PROJECT_ENVIRONMENT=$HOME/workspace/native_venv

# Install nki + neuronx-cc from local wheels
uv pip install $HOME/workspace/nki_wheels/nki-0.4.0*-cp312-cp312-linux_x86_64.whl
uv pip install $HOME/workspace/neuronx_cc_wheels/neuronx_cc-2.*-cp312-cp312-linux_x86_64.whl

# Install torch_neuronx editable (brings correct torch 2.11.0+cpu and torch_mlir)
cd $HOME/workspace/torch_neuron_eager
uv pip install -e .[dev]

echo "===== Step 7: Verify ====="
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch_neuronx; print('torch_neuronx:', getattr(torch_neuronx, '__version__', 'installed'))"
python -c "import nki; print('nki:', getattr(nki, '__version__', 'installed'))"
echo
neuron-ls | head -20
echo
echo "===== Beta 3 setup complete ====="
