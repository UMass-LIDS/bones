#!/bin/bash

function x264_encode() {
    local bitrate=$1
    local width=$2
    local height=$3
    local input_video=$4
    local method=$5

    ffmpeg -i ${input_video} -an -c:v libx264 -preset veryfast -b:v ${bitrate}k -maxrate $((bitrate * 2))k -bufsize $((bitrate * 4))k -x264opts "keyint=120:min-keyint=120:scenecut=0:no-scenecut" -pass 1 -r 30 -vf scale=w=${width}:h=${height}:flags=${method} output_${bitrate}k.mp4
}

function segment_video_m4s() {
    local bitrate1=$1
    local bitrate2=$2
    local bitrate3=$3
    local bitrate4=$4
    local bitrate5=$5
    local bitrate6=$6

    MP4Box -dash 4000 -frag 4000 -rap -bs-switching no -profile dashavc264:live -url-template output_${bitrate1}k.mp4:id="1440p" output_${bitrate2}k.mp4:id="1080p" output_${bitrate3}k.mp4:id="720p" output_${bitrate4}k.mp4:id="480p" output_${bitrate5}k.mp4:id="360p" output_${bitrate6}k.mp4:id="240p" -segment-name '$RepresentationID$/segment_$Number$' -out multi_resolution.mpd

    mv ./output_${bitrate1}* 1440p
    mv ./output_${bitrate2}* 1080p
    mv ./output_${bitrate3}* 720p
    mv ./output_${bitrate4}* 480p
    mv ./output_${bitrate5}* 360p
    mv ./output_${bitrate6}* 240p
}


function segment_video_mp4 {
  local folder=$1
  local bitrate=$2

  cd ${folder} || return
  mkdir segments_m4s
  mv ./segment_* segments_m4s

  ffmpeg -i output_${bitrate}k.mp4 -c copy -map 0 -segment_time 4 -f segment -reset_timestamps 1 -segment_start_number 1 segment_%d.mp4

  mkdir segments_mp4
  mv ./segment_* segments_mp4

  cd ..
}



OPTIND=1

while getopts ":i:" opt; do
    case "$opt" in
    i)
        input=$OPTARG
        ;;
    *)
        echo "Usage: $0 -i [input_file]" 1>&2; exit 1;
        ;;
    esac
done

if [ $OPTIND -eq 1 ]; then echo "Usage: $0 -i [input_file]" 1>&2; exit 1; fi


#command -v x264 >/dev/null 2>&1 || { echo >&2 "x264 not installed"; exit 1;}
#command -v MP4 >/dev/null 2>&1 || { echo >&2 "MP4 not installed"; exit 1;}


echo "x264 encoding"
x264_encode "6640" "2560" "1440" $input bicubic
x264_encode "4580" "1920" "1080" $input bicubic
x264_encode "2440" "1280" "720" $input bicubic
x264_encode "1060" "854" "480" $input bicubic
x264_encode "740" "640" "360" $input bicubic
x264_encode "340" "426" "240" $input bicubic

base_dir=$(basename $input .mp4)

echo "mkdir input dir"
mkdir -p ./1440p
mkdir -p ./1080p
mkdir -p ./720p
mkdir -p ./480p
mkdir -p ./360p
mkdir -p ./240p

segment_video_m4s "6640" "4580" "2440" "1060" "740" "340"

segment_video_mp4 "240p" "340"
segment_video_mp4 "360p" "740"
segment_video_mp4 "480p" "1060"
segment_video_mp4 "720p" "2440"
segment_video_mp4 "1080p" "4580"
segment_video_mp4 "1440p" "6640"

