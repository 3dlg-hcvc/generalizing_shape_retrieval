import json
from tqdm import tqdm


data_list = json.load(open("./data/pix3d/pix3d.json"))

pix3d_pose = {}

for i in tqdm(range(len(data_list))):
    img_name = data_list[i]['img'].replace('img/', '').split('.')[0]
    pix3d_pose[img_name] = {
        'img_name': img_name,
        'img_size': data_list[i]['img_size'], # (w, h)
        'rot_mat': data_list[i]['rot_mat'],
        'trans_mat': data_list[i]['trans_mat'],
        'cam_position': data_list[i]['cam_position'], # look at (0,0,0)
        'cam_look_at': [0,0,0],
        'cam_inp_rot': data_list[i]['inplane_rotation'],
        'focal_length': data_list[i]['focal_length'], # mm
        'sensor_width': 32, # mm
        'bbox': data_list[i]['bbox']
    }
    
with open("./data/pix3d/pix3d_pose.json", "w") as f:
    json.dump(pix3d_pose, f)