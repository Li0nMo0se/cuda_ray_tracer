#!/bin/bash
ffmpeg -pattern_type glob -framerate 20 -i "*scene*.ppm" output.mp4
