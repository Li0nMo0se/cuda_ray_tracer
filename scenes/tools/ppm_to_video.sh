#!/bin/bash
ffmpeg -pattern_type glob -framerate 20 -i "output/*.ppm" -crf 0 $1
