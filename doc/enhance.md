# Data Collection

To reproduce our data collection procedure, please follow the instructions below.

Install [PyTorch](https://pytorch.org/get-started/locally/), 
[FFmpeg](https://ffmpeg.org/), 
[MP4Box](https://github.com/gpac/gpac/wiki/MP4Box).

Download the ground truth 2160p 60-fps Big Buck Bunny video from [here](http://bbb3d.renderfarming.net/download.html)
([backup link](https://www.dropbox.com/scl/fi/rkwj1ivyw55tn0zpuzm6w/bbb_2160p_60fps.mp4?rlkey=jw3xtsc636kotc6xo9ymgbrlr&dl=0)).
Put it into the data/bbb/ directory.

Run the following command to segment the video into m4s segments (for streaming) and mp4 segments (for enhancement):

```
bash segment_1080p.sh -i bbb_2160p_60fps.mp4
```

The resulting data structure should look like this:

```
bbb_2160p_60fps.mp4
    1080p/
        output_4800k.mp4
        segments_m4s/
            segment_.mp4
            segment_1.m4s
            segment_2.m4s
            ...
        segments_mp4/
            segment_1.mp4
            segment_2.mp4
            ...
    720p/
    480p/
    360p/
    240p/
```

## NAS

Then, we train and evaluate the NAS-MDSR model on these video segments using codes modified from 
[the official implementaion](https://github.com/kaist-ina/NAS_public).



Train the super-resolution models in different quality levels using the script in the enhance/NAS/ forlder :

```
python train_nas_awdnn.py --quality low
python train_nas_awdnn.py --quality medium
python train_nas_awdnn.py --quality high
python train_nas_awdnn.py --quality ultra
```

Evaluate model inference time using:

```
python test_nas_video_runtime.py --quality low
python test_nas_video_runtime.py --quality medium
python test_nas_video_runtime.py --quality high
python test_nas_video_runtime.py --quality ultra
```

Then, enhance the low-resolution video segments using the trained models:

```
python enhance_mp4.py --quality low
python enhance_mp4.py --quality medium
python enhance_mp4.py --quality high
python enhance_mp4.py --quality ultra
```

Finally, extract video metadata, base video quality, and enhanced video quality using the script [here](../data/bbb/measure.py):

```
python measure.py
```


## IMDN

We adopt the [official implementation](https://github.com/Zheng222/IMDN/tree/master)
of the IMDN model. And we provide the pre-trained model [here](../enhance/IMDN/checkpoints).

To train the model, move to the enhance/IMDN/ folder and run:
```
python train.py
```

To evaluate, run:
```
python enhance_mp4.py --quality low
python enhance_mp4.py --quality medium
python enhance_mp4.py --quality high
python enhance_mp4.py --quality ultra
```

and then use the same script as NAS:
```
python measure.py
```

To measure the computation latency, run:
```
python measure_latency.py
```