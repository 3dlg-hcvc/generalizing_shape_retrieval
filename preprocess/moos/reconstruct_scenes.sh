#!/usr/bin/bash

n=1
data_dir="./data/moos/scenes"
out_dir="./data/moos/recon_scenes"
src_path="./preprocess/moos/reconstruct_scene.py"
scene_names="./data/moos/scene_names.txt"

mkdir -p $out_dir

parallel -j $n --eta "
    python3 $src_path \
    --data_dir $data_dir \
    --out_dir $out_dir \
    --scene_name {1} > /dev/null 2>&1" :::: $scene_names
