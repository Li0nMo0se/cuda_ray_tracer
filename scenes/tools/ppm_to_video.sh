#!/bin/bash
ffmpeg -pattern_type glob -framerate 20 -i "output/*.ppm" output.mp4
