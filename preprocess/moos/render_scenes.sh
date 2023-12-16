#!/usr/bin/bash

n=1
out_dir="./data/moos/scenes"
obj_source="3dfuture"
src_path="./preprocess/moos/render_sample_scene.py"
scene_ids="./data/moos/scene_ids.txt"

mkdir -p $out_dir

parallel -j $n --eta "
    python3 $src_path \
    --out_dir $out_dir \
    --scene_id {1} \
    --obj_source $obj_source > /dev/null 2>&1" :::: $scene_ids
