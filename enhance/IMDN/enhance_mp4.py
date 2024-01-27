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


def enhance_one_segment(segment_path, output_path, model, scale, device="cuda"):
    frames = read_frames(segment_path, scale)
    writer = imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=30,
                                codec='libx264', quality=10, pixelformat='yuv444p', macro_block_size=12)

    for frame in frames:
        im_input = frame / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        im_input = im_input.to(device)

        with torch.no_grad():
            output = model(im_input)
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

            writer.append_data(output)

    writer.close()
    return


def enhance(opt):
    resolution_list = [240, 360, 480]
    res2scale = {240: 4, 360: 3, 480: 2}

    print("Model", opt.model)

    for resolution in resolution_list:
        scale = res2scale[resolution]
        print("resolution", resolution)

        if opt.model == "IMDN_ultra":
            model = architecture.IMDN(upscale=scale)
        elif opt.model == "IMDN_high":
            model = architecture.IMDN(upscale=scale, nf=32)
        elif opt.model == "IMDN_medium":
            model = architecture.IMDN_RTC(upscale=scale)
        elif opt.model == "IMDN_low":
            model = architecture.IMDN_RTC(upscale=scale, num_modules=3, nf=6)
        else:
            raise NotImplementedError

        model_path = os.path.join(opt.checkpoint, '{}_{}_x{}.pth'.format(opt.model, opt.dataset, scale))
        model_dict = utils.load_state_dict(model_path)
        model.load_state_dict(model_dict, strict=True)
        model.eval()
        model = model.cuda()

        video_dir = os.path.join(opt.data, '{}p'.format(resolution), "segments_mp4")
        output_dir = os.path.join(opt.data, '{}p'.format(resolution), "{}_{}".format(opt.model, opt.dataset))
        os.makedirs(output_dir, exist_ok=True)

        segment_names = os.listdir(video_dir)
        for segment_name in segment_names:
            if segment_name.endswith('.mp4'):
                print(segment_name)
                segment_path = os.path.join(video_dir, segment_name)
                output_path = os.path.join(output_dir, segment_name)
                enhance_one_segment(segment_path, output_path, model, scale)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='IMDN_medium')
    parser.add_argument("--dataset", type=str, default='DIV2K')
    parser.add_argument("--checkpoint", type=str, default='checkpoints/')
    parser.add_argument("--data", type=str, default="../../data/bbb/")
    opt = parser.parse_args()

    enhance(opt)
