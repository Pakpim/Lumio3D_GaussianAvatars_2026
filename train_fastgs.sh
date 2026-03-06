#!/bin/bash

DATASET_PATH=${1:-"/rpool/data/icy/dataset/bird_v01"}
OUTPUT_PATH=${2:-"./output/fastgs_$(basename $DATASET_PATH)_$(date +"%s")"}

python train.py --source_path "$DATASET_PATH" --model_path "$OUTPUT_PATH"
