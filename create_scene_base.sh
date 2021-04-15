#!/bin/bash

file=$1

    # empty first
    echo "" > ${file}

    # Camera
    echo "Camera (0,1,-2) (0,1,0) (0,0,1) 1 45 45" >> ${file}

    # Light
    echo "PointLight (-4,4,0) 1" >> ${file}

    # Texture
    echo "UniformTexture red (255,0,0) 1 0 0" >> ${file}
    echo "UniformTexture green (0,255,0) 1 0 0" >> ${file}
    echo "UniformTexture blue (0,0,255) 1 0 0" >> ${file}

    # Plan
    echo "UniformTexture ground_texture (100,100,100) 1 0.5 1" >> ${file}
    echo "Plan (0,0,0) (0,1,0) ground_texture" >> ${file}

    for i in $(seq 120); do
	if [ $((i % 3)) -eq 0 ]; then
		echo "Sphere ($((i - 60)),1,0) 0.5 red (0,0,0.2)" >> ${file}
	elif [ $((i % 3)) -eq  1 ]; then
		echo "Sphere ($((i - 60)),1,0) 0.5 green (0,0,0.2)" >> ${file}
        else
	        echo "Sphere ($((i - 60)),1,0) 0.5 blue (0,0,0.2)" >> ${file}
	fi
    done
