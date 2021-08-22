
import argparse
from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.

import numpy as np

from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

from subprocess import Popen, PIPE
import re

from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Create the parser
vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

# Add the arguments
vq_parser.add_argument("-o",    "--output", type=str, help="Output file", default="output.png", dest='output')
vq_parser.add_argument("-vl",   "--video_length", type=float, help="Video length in seconds", default=10, dest='video_length')
vq_parser.add_argument("-la",    "--video_last", type=int, help="Video last frame", dest='video_last')

# Execute the parse_args() method
args = vq_parser.parse_args()

init_frame = 1  # Initial video frame
last_frame = args.video_last
length = args.video_length  # Desired time of the video in seconds

min_fps = 10
max_fps = 60

total_frames = last_frame - init_frame

frames = []
print('Generating video...')
for i in range(init_frame, last_frame):
    frames.append(Image.open("./steps/" + str(i) + '.png'))

# fps = last_frame/10
fps = np.clip(total_frames / length, min_fps, max_fps)
output_file = re.compile('\.png$').sub('.mp4', args.output)
p = Popen(['ffmpeg',
           '-y',
           '-f', 'image2pipe',
           '-vcodec', 'png',
           '-r', str(fps),
           '-i',
           '-',
           '-vcodec', 'libx264',
           '-r', str(fps),
           '-pix_fmt', 'yuv420p',
           '-crf', '17',
           '-preset', 'veryslow',
           output_file], stdin=PIPE)
for im in tqdm(frames):
    im.save(p.stdin, 'PNG')
p.stdin.close()
p.wait()