#!/bin/bash

function x264_encode() {
    local bitrate=$1
    local width=$2
    local height=$3
    local input_video=$4
    local method=$5

    ffmpeg -i ${input_video} -an -c:v libx264 -preset veryfast -b:v ${bitrate}k -maxrate $((bitrate * 2))k -bufsize $((bitrate * 4))k -x264opts "keyint=120:min-keyint=120:scenecut=0:no-scenecut" -pass 1 -r 30 -vf scale=w=${width}:h=${height}:flags=${method} output_${height}p.mp4
#    ~/Projects/enhancement/ffmpeg/build/bin/ffmpeg -i ${input_video} -an -c:v libx264 -preset veryfast -b:v ${bitrate}k -maxrate $((bitrate * 2))k -bufsize $((bitrate * 4))k -x264opts "keyint=120:min-keyint=120:scenecut=0:no-scenecut" -pass 1 -r 30 -vf scale=w=${width}:h=${height}:flags=${method} output_${bitrate}k.mp4

}

function segment_video_m4s() {
    MP4Box -dash 4000 -frag 4000 -rap -bs-switching no -profile dashavc264:live -url-template output_240p.mp4:id="0" output_360p.mp4:id="1" output_480p.mp4:id="2" output_720p.mp4:id="3" output_1080p.mp4:id="4" -segment-name 'stream$RepresentationID$/segment_$Number$' -out bbb.mpd
}


input="bbb_2160p_60fps.mp4"

x264_encode "4800" "1920" "1080" $input bicubic
x264_encode "2400" "1280" "720" $input bicubic
x264_encode "1200" "854" "480" $input bicubic
x264_encode "800" "640" "360" $input bicubic
x264_encode "400" "426" "240" $input bicubic

mkdir -p ./stream0
mkdir -p ./stream1
mkdir -p ./stream2
mkdir -p ./stream3
mkdir -p ./stream4

segment_video_m4s
