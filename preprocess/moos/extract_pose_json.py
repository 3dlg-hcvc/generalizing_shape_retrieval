import os
import json
import pandas as pd
from tqdm import tqdm

data_dir = "./data/moos/scenes"
df = pd.read_csv("./data/moos/moos_annotation_all.txt")

moos_pose = {}
for i, row in tqdm(df.iterrows()):
    full_view_id = row.full_view_id
    scene_id = row.scene_id
    scene_name = f"scene_{scene_id:05d}"
    obj_scene_id = row.obj_scene_id
    view_id = row.view_id
    scene_info_path = os.path.join(data_dir, scene_name, "scene.json")
    scene_info = json.load(open(scene_info_path))
    moos_pose[full_view_id] = {
        "full_view_id": full_view_id,
        "img_size": scene_info["metadata"]["img_size"],
        "rot_mat": scene_info["objects"][obj_scene_id]["rot_mat3d"],
        "trans_mat": scene_info["objects"][obj_scene_id]["trans3d"],
        "cam_dist": scene_info["cameras"]["dist"],
        "cam_azim": scene_info["cameras"]["azims"][view_id],
        "cam_elev": scene_info["cameras"]["elevs"][view_id],
        "cam_look_at": scene_info["cameras"]["look_at"],
        "focal_length": scene_info["cameras"]['focal_length'],
        "sensor_width": scene_info["cameras"]['sensor_width'],
    }
    
with open("./data/moos/moos_pose.json", "w") as f:
    json.dump(moos_pose, f)