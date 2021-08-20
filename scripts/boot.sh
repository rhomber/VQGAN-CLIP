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

chmod +x ./download_models.sh
./download_models.sh

