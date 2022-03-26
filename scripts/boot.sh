#!/bin/bash

set -o pipefail

. /opt/conda/etc/profile.d/conda.sh
conda activate vqgan

cd /app

if [ ! -d './CLIP' ]; then
  git clone https://github.com/openai/CLIP
fi

if [ ! -d './taming-transformers' ]; then
  git clone https://github.com/CompVis/taming-transformers.git
fi

if [ ! -d './clipit' ]; then
  git clone https://github.com/mkualquiera/clipit.git
fi

chmod +x ./download_models.sh
./download_models.sh

ulimit -n 1048576
ulimit -Sn 1048576
