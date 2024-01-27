import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture
import imageio
import time


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="IMDN_ultra")
parser.add_argument("--checkpoint", type=str, default="",
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--scale", type=int, default=3,
                    help='upscaling factor')
opt = parser.parse_args()

print(opt)

def read_frames(video_file, scale):
    frames = []
    cap = cv2.VideoCapture(video_file)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1920 // scale, 1080 // scale), interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames



cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

if opt.model == "IMDN_ultra":
    model = architecture.IMDN(upscale=opt.scale)
elif opt.model == "IMDN_high":
    model = architecture.IMDN(upscale=opt.scale, nf=32)
elif opt.model == "IMDN_medium":
    model = architecture.IMDN_RTC(upscale=opt.scale)
elif opt.model == "IMDN_low":
    model = architecture.IMDN_RTC(upscale=opt.scale, num_modules=3, nf=6)
else:
    raise NotImplementedError

if opt.checkpoint != "":
    model_dict = utils.load_state_dict(opt.checkpoint)
    model.load_state_dict(model_dict, strict=True)
model = model.to(device)

scale2res = {4:"240p", 3:"360p", 2:"480p"}
res = scale2res[opt.scale]

frames = read_frames('../segment_{}.mp4'.format(res), opt.scale)
writer = imageio.get_writer("imdn_{}.mp4".format(res), format='FFMPEG', mode='I', fps=30,
                                codec='libx264', quality=10, pixelformat='yuv444p', macro_block_size=12)

for frame_idx in range(len(frames)):
    frame = frames[frame_idx]
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame[np.newaxis, ...]
    frame = torch.from_numpy(frame).float()
    frames[frame_idx] = frame

start_time = None
outputs = []

for frame_idx in range(len(frames)):
    if frame_idx == 5:
        start_time = time.time()

    with torch.no_grad():
        im_input = frames[frame_idx].to(device)
        out = model(im_input)
        output = out.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        outputs.append(output)

torch.cuda.synchronize()
end_time = time.time()
print("Average time", (end_time - start_time) / (len(frames) - 5))

if not opt.checkpoint:
    exit()

for output in outputs:
    if output.ndim == 3:
        output = np.transpose(output, (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    writer.append_data(output)
writer.close()
