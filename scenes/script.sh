#!/bin/bash

mkdir -p -v video;

for i in $(seq 60); do
    if ((i < 10)); then
        file="video/0${i}.scene"
    else
        file="video/${i}.scene"
    fi
    # empty first
    echo "" > ${file}

    # Camera
    echo "Camera (0,1,-2) (0,1,0) (0,0,1) 1 45 45" >> ${file}

    # Light
    echo "PointLight (-4,4,0) 1" >> ${file}

    # Sphere 1
    echo "UniformTexture red (255,0,0) 1 0 0" >> ${file}
    if ((i % 2)); then
        echo "Sphere (0,1,$((i / 2)).5) 1 red" >> ${file}
    else
        echo "Sphere (0,1,$((i / 2))) 1 red" >> ${file}
    fi

    # Sphere 2
    echo "UniformTexture green (0,255,0) 1 0 0" >> ${file}
    echo "Sphere (-6,1,${i}) 1 green" >> ${file}

    # Sphere 3
    echo "UniformTexture blue (0,0,255) 1 0 0" >> ${file}
    echo "Sphere (6,1,$((i * 2))) 1 blue" >> ${file}

    # Ground
    echo "UniformTexture ground_texture (100,100,100) 1 0 0" >> ${file}
    echo "Plan (0,0,0) (0,1,0) ground_texture" >> ${file}



done
