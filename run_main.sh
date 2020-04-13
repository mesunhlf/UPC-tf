#!/bin/bash
PREFIX="person"

env CUDA_VISIBLE_DEVICES="0" python main.py
        2> errors_${PREFIX}.txt \
        | tee results/${PREFIX}/params.txt

