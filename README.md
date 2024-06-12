# BONES

This is the official code repository for the paper: 
[BONES: Near-Optimal Neural-Enhanced Video Streaming](https://arxiv.org/abs/2310.09920) .
Authors: Lingdong Wang, Simran Singh, Jacob Chakareski, Mohammad Hajiesmaili, Ramesh K. Sitaraman.
Pulication: ACM SIGTMETICS 2024.

## Simulation Environment

### Prerequisites

numpy

### Quick Start

Move to the control/folder, and then run our BONES algorithm with:

```bash
python bones.py --monitor --verbose
```


### Data

We provide the video metadata and the enhancement performance as JSON files [here](data/bbb/).

Refer to [this document](doc/enhance.md) if you want to reproduce our data collection procedure. 


### Reproduction

To reproduce the results in our paper, move to the experiment/ folder and then run:

```bash
python run_all.py
```

## Prototype System 

### Prerequisites

asyncio,
PyTorch, [VPF](https://github.com/NVIDIA/VideoProcessingFramework), OpenGL

This system requires a GPU with CUDA support. It has only been tested on Linux.

### Data

Download the sample DASH video segments from [here](https://www.dropbox.com/scl/fi/9r2dyjmc05ch6gm1u6dxf/video_mp4box.zip?rlkey=72rnalr0cex3nc1i7u1u70333&dl=0).
Unzip the file and overwrite the player/data/video_mp4box/ folder with it.

If you want to reproduce the generation of video segments, move to player/data/video_mp4box/ and run:

```bash
bash segment_1080p_mp4box.sh
```


### Usage

Move to the player/ folder, and then run our prototype system with:

```
python play.py -i data/video_mp4box/bbb.mpd -t data/3Glogs/report.2010-09-13_1003CEST.json
```

The player runs in the headless mode by default.
To enable the graphical renderer, run:

```
python play.py -i data/video_mp4box/bbb.mpd -t data/3Glogs/report.2010-09-13_1003CEST.json --mod_renderer opengl
```

Please note that this implementation is a research prototype favoring flexibility over efficiency, 
so the graphical mode may suffer from low frame rate or rendering artifacts.
To achieve better performance, we recommend using the headless mode, running the graphical mode with powerful GPU, or decreasing the display_width/display_height in the [configuration file](player/istream_player/config/config.py).


## Acknowledgement

Our implementation of the simulation environment is based on [Sabre](https://github.com/UMass-LIDS/sabre).

Our implementation of the prototype system is based on [iStream](https://github.com/NetMedia-Sys-Lab/istream-player).


## Citation

If you find our work helpful, please consider citing:

```
@inproceedings{10.1145/3652963.3655047,
author = {Wang, Lingdong and Singh, Simran and Chakareski, Jacob and Hajiesmaili, Mohammad and Sitaraman, Ramesh K.},
title = {BONES: Near-Optimal Neural-Enhanced Video Streaming},
year = {2024},
isbn = {9798400706240},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3652963.3655047},
doi = {10.1145/3652963.3655047},
booktitle = {Abstracts of the 2024 ACM SIGMETRICS/IFIP PERFORMANCE Joint International Conference on Measurement and Modeling of Computer Systems},
pages = {61–62},
numpages = {2},
keywords = {adaptive bitrate streaming, lyapunov optimization, neural enhancement, super-resolution},
location = {, Venice, Italy, },
series = {SIGMETRICS/PERFORMANCE '24}
}
```
