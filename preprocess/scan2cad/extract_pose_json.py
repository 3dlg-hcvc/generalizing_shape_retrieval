import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import quaternion

data_dir = "/path_to_roca_data"
inst_path_temp = "/path_to_roca_data/Rendering/{scene_id}/instance/{img_id}.png"

image_alignments = json.load(open(f"{data_dir}/Dataset/scan2cad_image_alignments.json"))
images = image_alignments["images"]
alignments = image_alignments["alignments"]

annotation = pd.read_csv("./data/scan2cad/scan2cad_annotation.txt")

scan2cad_pose = {}

def make_M_from_tqs(t: list, q: list, s: list) -> np.ndarray:
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    return M


for full_img_id in tqdm(alignments):
    img_info = images[full_img_id]
    obj_infos = alignments[full_img_id]
    scene_id, img_id = full_img_id.split('/')
    inst_path = inst_path_temp.format(scene_id=scene_id, img_id=img_id)
    
    inst = Image.open(inst_path)
    inst_np = np.array(inst)
    inst_indices = np.unique(inst_np)
    if len(inst_indices) == 0: continue
    
    for i, idx in enumerate(inst_indices):
        if idx == 0: continue
        
        full_mask_id = f"{full_img_id}.{i}"
        obj = obj_infos[idx-1]
        M = make_M_from_tqs(obj['t'], obj['q'], obj['s']).astype(float)
        img_size = (img_info['width'], img_info['height'])
    
        scan2cad_pose[full_mask_id] = {
            'mask_name': full_mask_id,
            'img_name': full_img_id,
            'img_size': img_size, # (w, h)
            'rot_mat': M[:3, :3].tolist(),
            'trans_mat': M[:3, 3].tolist(),
            'cam_position': [0,0,0], # look at (0,0,0)
            'cam_look_at': [0,0,-1],
            'cam_intrinsics': img_info['intrinsics'], # mm
        }
    
with open("./data/scan2cad/scan2cad_pose.json", "w") as f:
    json.dump(scan2cad_pose, f)