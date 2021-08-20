#!/bin/bash

conda init bash
bash
conda activate vqgan

cd /app

if [ ! -d './CLIP' ]; then
  git clone https://github.com/openai/CLIP
fi

if [ ! -d './taming-transformers' ]; then
  git clone https://github.com/CompVis/taming-transformers.git
fi

./download_models.sh
