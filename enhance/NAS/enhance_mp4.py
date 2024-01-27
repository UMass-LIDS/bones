import json
import os
import numpy as np
import torch
from option import opt
from model import MultiNetwork
import template
import utility as util
import cv2
import imageio


def get_resolution(quality):
    assert quality in [0, 1, 2, 3]

    if quality == 3:
        t_w = 1920
        t_h = 1080
    elif quality == 2:
        t_w = 960
        t_h = 540
    elif quality == 1:
        t_w = 640
        t_h = 360
    elif quality == 0:
        t_w = 480
        t_h = 270
    return (t_h, t_w)


def enhance_one_segment(segment_path, output_path, video_info, model, inference_idx, fps=30):
    # enhance one mp4 segment

    t_h, t_w = get_resolution(video_info.quality)
    target_scale = int(1080 / t_h)
    target_height = t_h

    results = []
    vc = cv2.VideoCapture(segment_path)

    while True:
        rval, frame = vc.read()
        if rval == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (t_w, t_h), interpolation=cv2.INTER_CUBIC)  # add bicubic resize
        input_tensor = torch.from_numpy(frame).cuda()

        with torch.no_grad():
            input_tensor = input_tensor.permute(2, 0, 1).half()
            input_tensor.div_(255)  # byte tensor/255
            input_tensor.unsqueeze_(0)

            output_tensor = model(input_tensor, inference_idx)
            output_tensor = output_tensor.data[0].permute(1, 2, 0)
            output_tensor = output_tensor * 255
            output_tensor = torch.clamp(output_tensor, 0, 255)
            results.append(output_tensor.cpu().numpy().astype(np.uint8))
    vc.release()

    writer = imageio.get_writer(output_path, format='FFMPEG', mode='I', fps=fps,
                                codec='libx264', quality=10, pixelformat='yuv444p', macro_block_size=12)
    for frame in results:
        writer.append_data(frame)
    writer.close()
    return


def enhance_one_quality():
    # hyper-parameters
    segment_fps = 30
    segment_size = 4
    scale_list = [4, 3, 2, 1]
    resolution_list = [240, 360, 480, 720]
    res2quality = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}

    # prepare model
    pretrained_path = os.path.join(opt.checkpoint_dir, 'epoch_{}.pth'.format(opt.test_num_epoch))

    model = MultiNetwork(template.get_nas_config(opt.quality))
    model = model.to(torch.device('cuda'))
    model = model.half().to('cuda')
    model.load_state_dict(torch.load(pretrained_path))

    # prepare data
    for resolution_idx in range(len(resolution_list)):
        resolution = resolution_list[resolution_idx]
        model.setTargetScale(scale_list[resolution_idx])
        inference_idx = max(0, len(model.getOutputNodes()) - 1)
        print("Inference idx", inference_idx)
        video_dir = os.path.join(opt.data_dir, opt.data_name, '{}p'.format(resolution), "segments_mp4")
        output_dir = os.path.join(opt.data_dir, opt.data_name, '{}p'.format(resolution), "nas_{}".format(opt.quality))
        os.makedirs(output_dir, exist_ok=True)
        video_info = util.videoInfo(segment_fps, segment_size, res2quality[resolution])
        segment_names = os.listdir(video_dir)
        for segment_name in segment_names:
            if segment_name.endswith('.mp4'):
                segment_path = os.path.join(video_dir, segment_name)
                output_path = os.path.join(output_dir, segment_name)
                enhance_one_segment(segment_path, output_path, video_info, model, inference_idx)
    return


if __name__ == '__main__':
    enhance_one_quality()