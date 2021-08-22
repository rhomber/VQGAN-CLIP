#!/bin/bash

set -o pipefail

mkdir -p /tmp
cd /tmp

apt update
apt install -y git curl wget vim ffmpeg

wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh
./Miniconda3-py38_4.10.3-Linux-x86_64.sh -b
rm -f ./Miniconda3-py38_4.10.3-Linux-x86_64.sh

cd /app

conda create --name vqgan python=3.9
conda init bash
. /opt/conda/etc/profile.d/conda.sh
source activate vqgan
echo "Python: $(python --version)"

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer

# cgd

git clone https://github.com/afiaka87/clip-guided-diffusion.git && cd clip-guided-diffusion
git clone https://github.com/afiaka87/guided-diffusion.git
pip3 install -e guided-diffusion
python setup.py install
