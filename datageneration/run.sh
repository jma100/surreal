#!/bin/bash

JOB_PARAMS=${1:-'--idx 0 --ishape 0 --stride 50'} # defaults to [0, 0, 50]

# SET PATHS HERE
FFMPEG_PATH=/home/jingweim/urop/ffmpeg/ffmpeg_build_sequoia_h264
X264_PATH=/home/jingweim/urop/ffmpeg/x264_build/
PYTHON2_PATH=/home/jingweim/urop/surreal/surrealenv/ # PYTHON 2
BLENDER_PATH=/home/jingweim/urop/blender-2.78b-linux-glibc219-x86_64
cd /home/jingweim/urop/surreal/datageneration

# BUNLED PYTHON
BUNDLED_PYTHON=${BLENDER_PATH}/2.78/python
export PYTHONPATH=${BUNDLED_PYTHON}/lib/python3.4:${BUNDLED_PYTHON}/lib/python3.4/site-packages
export PYTHONPATH=${BUNDLED_PYTHON}:${PYTHONPATH}

# FFMPEG
export LD_LIBRARY_PATH=${FFMPEG_PATH}/lib:${X264_PATH}/lib:${LD_LIBRARY_PATH}
export PATH=${FFMPEG_PATH}/bin:${PATH}

# ### DEBUG
# $BLENDER_PATH/blender -P main_part1.py -- ${JOB_PARAMS}

# ### RUN PART 1  --- Uses python3 because of Blender
$BLENDER_PATH/blender -b -t 1 -P main_part1.py -- ${JOB_PARAMS}

# ### RUN PART 2  --- Uses python2 because of OpenEXR
# PYTHONPATH="" ${PYTHON2_PATH}/bin/python2.7 main_part2.py ${JOB_PARAMS}
