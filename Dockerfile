FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

WORKDIR /app

#COPY ./download_models.sh /app/download_models.sh
#COPY ./generate.py /app/generate.py
#COPY ./requirements.txt /requirements.txt
COPY . .

RUN chmod +x ./download_models.sh

RUN apt update
RUN apt install -y git curl wget

RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install ftfy regex tqdm omegaconf pytorch-lightning IPython kornia imageio imageio-ffmpeg einops torch_optimizer
#RUN python -m pip install -r /requirements.txt

RUN git clone https://github.com/openai/CLIP
RUN git clone https://github.com/CompVis/taming-transformers.git
