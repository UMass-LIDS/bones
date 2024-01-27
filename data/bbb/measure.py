import subprocess
import re
import numpy as np
import json
import os
from tqdm import tqdm

def run_ffmpeg(ground_truth, input_video, metric, rescale=True):

    if metric == "psnr":
        metric_cmd = "[distorted][reference]psnr"
    elif metric == "ssim":
        metric_cmd = "[distorted][reference]ssim"
    elif metric == "vmaf":
        metric_cmd = "[distorted][reference]libvmaf=n_threads=4"
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")

    if rescale:
        command = f'ffmpeg -i {ground_truth} -i {input_video} -lavfi "[1:v]scale=1920:1080[scaled];[0:v]setpts=PTS-STARTPTS[reference];[scaled]setpts=PTS-STARTPTS[distorted];{metric_cmd}" -f null -'
    else:
        command = f'ffmpeg -i {ground_truth} -i {input_video} -lavfi "[0:v]setpts=PTS-STARTPTS[reference];[1:v]setpts=PTS-STARTPTS[distorted];{metric_cmd}" -f null -'
    output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, text=True)

    match = None
    if metric == "psnr":
        match = re.search(r'average:([\d\.]+)', output)
    elif metric == "ssim":
        match = re.search(r'All:([\d\.]+)', output)
    elif metric == "vmaf":
        match = re.search(r"VMAF score: ([\d.]+)", output)

    if match:
        return float(match.group(1))
    else:
        raise RuntimeError(f"Could not find {metric} score in ffmpeg output")


def measure_base_quality():
    input_resolutions = [240, 360, 480, 720]
    target_resolution = 1080
    num_segments = 159
    psnr_scores = np.zeros((len(input_resolutions), num_segments))
    ssim_scores = np.zeros((len(input_resolutions), num_segments))
    vmaf_scores = np.zeros((len(input_resolutions), num_segments))

    print("Input resolutions:", input_resolutions)
    print("Target resolution:", target_resolution)
    print("Number of segments:", num_segments)

    for resolution_idx in range(len(input_resolutions)):
        input_resolution = input_resolutions[resolution_idx]
        print(f"Input resolution: {input_resolution}p")

        for segment_idx in range(1, num_segments + 1):
            input_video = f'{input_resolution}p/segments_mp4/segment_{segment_idx}.mp4'
            ground_truth = f'{target_resolution}p/segments_mp4/segment_{segment_idx}.mp4'
            psnr_score = run_ffmpeg(ground_truth, input_video, metric="psnr",rescale=True)
            psnr_scores[resolution_idx, segment_idx-1] = psnr_score
            ssim_score = run_ffmpeg(ground_truth, input_video, metric="ssim",rescale=True)
            ssim_scores[resolution_idx, segment_idx-1] = ssim_score
            vmaf_score = run_ffmpeg(ground_truth, input_video, metric="vmaf",rescale=True)
            vmaf_scores[resolution_idx, segment_idx-1] = vmaf_score

        print(f"Average PSNR score: {sum(psnr_scores[resolution_idx]) / len(psnr_scores[resolution_idx])}")
        print(f"Average SSIM score: {sum(ssim_scores[resolution_idx]) / len(ssim_scores[resolution_idx])}")
        print(f"Average VMAF score: {sum(vmaf_scores[resolution_idx]) / len(vmaf_scores[resolution_idx])}")

    return psnr_scores, ssim_scores, vmaf_scores


def measure_enhanced_quality(methods, input_resolutions):
    target_resolution = 1080
    num_segments = 159
    psnr_scores = np.zeros((len(input_resolutions), len(methods), num_segments))
    ssim_scores = np.zeros((len(input_resolutions), len(methods), num_segments))
    vmaf_scores = np.zeros((len(input_resolutions), len(methods),  num_segments))

    print("Input resolutions:", input_resolutions)
    print("Target resolution:", target_resolution)
    print("Number of segments:", num_segments)

    for resolution_idx in range(len(input_resolutions)):
        input_resolution = input_resolutions[resolution_idx]
        print(f"Input resolution: {input_resolution}p")

        for method_idx in range(len(methods)):
            method = methods[method_idx]
            print(f"Method: {method}")

            for segment_idx in range(1, num_segments + 1):
                input_video = f'{input_resolution}p/{method}/segment_{segment_idx}.mp4'
                ground_truth = f'{target_resolution}p/segments_mp4/segment_{segment_idx}.mp4'
                psnr_score = run_ffmpeg(ground_truth, input_video, metric="psnr", rescale=False)
                psnr_scores[resolution_idx][method_idx][segment_idx - 1] = psnr_score
                ssim_score = run_ffmpeg(ground_truth, input_video, metric="ssim", rescale=False)
                ssim_scores[resolution_idx][method_idx][segment_idx - 1] = ssim_score
                vmaf_score = run_ffmpeg(ground_truth, input_video, metric="vmaf", rescale=False)
                vmaf_scores[resolution_idx][method_idx][segment_idx - 1] = vmaf_score

            print(f"Average PSNR score: {sum(psnr_scores[resolution_idx][method_idx]) / len(psnr_scores[resolution_idx][method_idx])}")
            print(f"Average SSIM score: {sum(ssim_scores[resolution_idx][method_idx]) / len(ssim_scores[resolution_idx][method_idx])}")
            print(f"Average VMAF score: {sum(vmaf_scores[resolution_idx][method_idx]) / len(vmaf_scores[resolution_idx][method_idx])}")


    # vmaf_scores.tolist()
    # print("VMAF scores")
    # vmaf_scores = json.dumps(vmaf_scores)
    # print(vmaf_scores)
    return psnr_scores, ssim_scores, vmaf_scores


def measure_video_info():
    resolutions = [240, 360, 480, 720, 1080]
    num_segments = 159
    result = {
        "segment_duration_ms": 4000,
        "bitrates_kbps": [0 for _ in range(len(resolutions))],
        "segment_sizes_bits": [[] for _ in range(num_segments)],
    }

    for resolution_idx in range(len(resolutions)):
        resolution = resolutions[resolution_idx]
        for segment_idx in range(1, num_segments + 1):

            segment_path = f'{resolution}p/segments_m4s/segment_{segment_idx}.m4s'
            segment_size = os.path.getsize(segment_path) * 8  # in bits
            result["segment_sizes_bits"][segment_idx - 1].append(segment_size)
            result["bitrates_kbps"][resolution_idx] += segment_size

    for resolution_idx in range(len(resolutions)):
        result["bitrates_kbps"][resolution_idx] /= num_segments * result["segment_duration_ms"]
        result["bitrates_kbps"][resolution_idx] = int(result["bitrates_kbps"][resolution_idx])

    json.dump(result, open("bbb_video_info.json", "w"), indent=4)
    return


def measure_nas():
    methods = ["nas_low", "nas_medium", "nas_high", "nas_ultra"]
    input_resolutions = [240, 360, 480, 720]

    result = {}
    psnr_base, ssim_base, vmaf_base = measure_base_quality()
    result["base_quality_vmaf"] = vmaf_base.tolist()
    result["base_quality_ssim"] = ssim_base.tolist()
    result["base_quality_psnr"] = psnr_base.tolist()
    psnr_enh, ssim_enh, vmaf_enh = measure_enhanced_quality(methods, input_resolutions)
    result["enhanced_quality_vmaf"] = vmaf_enh.tolist()
    result["enhanced_quality_ssim"] = ssim_enh.tolist()
    result["enhanced_quality_psnr"] = psnr_enh.tolist()
    json.dump(result, open("nas_quality.json", "w"), indent=4)
    return


def measure_imdn_div():
    methods = ["IMDN_low_DIV2K", "IMDN_medium_DIV2K", "IMDN_high_DIV2K", "IMDN_ultra_DIV2K"]
    input_resolutions = [240, 360, 480]
    result = {}
    psnr_base, ssim_base, vmaf_base = measure_base_quality()
    result["base_quality_vmaf"] = vmaf_base.tolist()
    result["base_quality_ssim"] = ssim_base.tolist()
    result["base_quality_psnr"] = psnr_base.tolist()
    psnr_enh, ssim_enh, vmaf_enh = measure_enhanced_quality(methods, input_resolutions)
    result["enhanced_quality_vmaf"] = vmaf_enh.tolist()
    result["enhanced_quality_ssim"] = ssim_enh.tolist()
    result["enhanced_quality_psnr"] = psnr_enh.tolist()
    json.dump(result, open("imdn_div_quality.json", "w"), indent=4)
    return

def measure_imdn_bbb():
    methods = ["IMDN_low_bbb", "IMDN_medium_bbb", "IMDN_high_bbb", "IMDN_ultra_bbb"]
    input_resolutions = [240, 360, 480]
    result = {}
    psnr_base, ssim_base, vmaf_base = measure_base_quality()
    result["base_quality_vmaf"] = vmaf_base.tolist()
    result["base_quality_ssim"] = ssim_base.tolist()
    result["base_quality_psnr"] = psnr_base.tolist()
    psnr_enh, ssim_enh, vmaf_enh = measure_enhanced_quality(methods, input_resolutions)
    result["enhanced_quality_vmaf"] = vmaf_enh.tolist()
    result["enhanced_quality_ssim"] = ssim_enh.tolist()
    result["enhanced_quality_psnr"] = psnr_enh.tolist()
    json.dump(result, open("imdn_bbb_quality.json", "w"), indent=4)
    return


def generate_nas_1080ti():
    fps = [
        [65.98, 65.39, 60.85, 54.46],
        [56.39, 50.94, 46.75, 54.40],
        [40.03, 35.03, 30.41, 44.50],
        [25.27, 21.99, 21.06, 28.65]
    ]

    file = json.load(open(os.path.join("nas_quality.json")))
    output_name = "nas_1080ti.json"
    result = dict()
    result["frame_rate"] = 30
    result["enhancement_fps"] = fps
    for key in file:
        result[key] = file[key]
    json.dump(result, open(os.path.join(output_name), "w"), indent=4)

    return


def generate_imdn_2080ti():
    fps = [
        [48.85, 44.50, 32.54, 22.55],
        [43.42, 38.65, 24.54, 15.39],
        [32.47, 28.67, 14.38, 8.00],
        [-1, -1, -1, -1]
    ]

    file = json.load(open(os.path.join("imdn_div_quality.json")))
    output_name = "imdn_div2k_2080ti.json"
    result = dict()
    result["frame_rate"] = 30
    result["enhancement_fps"] = fps
    for key in file:
        result[key] = file[key]
    json.dump(result, open(os.path.join(output_name), "w"), indent=4)

    file = json.load(open(os.path.join("imdn_bbb_quality.json")))
    output_name = "imdn_bbb_2080ti.json"
    result = dict()
    result["frame_rate"] = 30
    result["enhancement_fps"] = fps
    for key in file:
        result[key] = file[key]
    json.dump(result, open(os.path.join(output_name), "w"), indent=4)
    return


def generate_imdn_3060ti():
    fps = [
        [133.30, 94.33, 41.67, 21.69],
        [113.79, 72.62, 27.02, 12.20],
        [81.43, 42.40, 13.61, 5.74],
        [-1, -1, -1, -1]
    ]

    file = json.load(open(os.path.join("imdn_div_quality.json")))
    output_name = "imdn_div2k_3060ti.json"
    result = dict()
    result["frame_rate"] = 30
    result["enhancement_fps"] = fps
    for key in file:
        result[key] = file[key]
    json.dump(result, open(os.path.join(output_name), "w"), indent=4)

    file = json.load(open(os.path.join("imdn_bbb_quality.json")))
    output_name = "imdn_bbb_3060ti.json"
    result = dict()
    result["frame_rate"] = 30
    result["enhancement_fps"] = fps
    for key in file:
        result[key] = file[key]
    json.dump(result, open(os.path.join(output_name), "w"), indent=4)
    return



if __name__ == '__main__':
    pass
    # measure_video_info()

    # measure_nas()
    # measure_imdn_div()
    # measure_imdn_bbb()

    # generate_nas_1080ti()
    generate_imdn_2080ti()
    generate_imdn_3060ti()