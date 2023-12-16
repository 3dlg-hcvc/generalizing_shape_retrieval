import warnings
warnings.filterwarnings('ignore')

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.renderer import BlendParams

from obj_util import (
    load_3dfuture_metadata, 
    load_3dfuture_obj, 
    create_meshes,
)
from render_util import (
    create_cameras,
    create_ligths,
    create_rasterizer,
    create_shader,
    create_renderer,
    write_scene_images
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

cats = ["chair", "bed", "table", "sofa"]
cat2num = {"chair": 1, "bed": 1, "table": 1, "sofa": 1}
shader_types = ["rgb", "instances", "depth", "normal"]


def reconstruct_one_scene(scene_name, data_dir, out_dir):
    scene_info_path = os.path.join(data_dir, f"{scene_name}/scene.json")
    scene_info = json.load(open(scene_info_path))

    scene_id = scene_info["id"]
    scene_name = f"scene_{scene_id:05d}"
    scene_dir = os.path.join(out_dir, scene_name)
    os.makedirs(os.path.join(scene_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "instances"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "objects"), exist_ok=True)

    _, obj_scales = load_3dfuture_metadata()
    obj_ids = [obj["id"] for obj in scene_info["objects"]]
    objs = []
    for i, obj_id in enumerate(obj_ids):
        obj_info = scene_info["objects"][i]
        obj = {"id": obj_id}
        obj["verts"], obj["faces"], obj["textures"] = load_3dfuture_obj(obj_id, obj_scales=obj_scales)
        trans3d = torch.tensor(obj_info["trans3d"]).type_as(obj["verts"])
        rot_mat3d = torch.tensor(obj_info["rot_mat3d"]).type_as(obj["verts"])
        obj["verts"] = (rot_mat3d @ obj["verts"].T).T + trans3d
        objs.append(obj)

    scene, mesh_list = create_meshes(objs, device=device)
    meshes = join_meshes_as_batch(mesh_list).to(device)

    img_size = scene_info["metadata"]["img_size"]
    lights = create_ligths(location=scene_info["lights"]["positions"], device=device)
    num_views = scene_info["metadata"]["num_views"]
    cameras, _, _ = create_cameras(dist=scene_info["cameras"]["dist"],
                                            at=(scene_info["cameras"]["look_at"],),
                                            img_size=img_size,
                                            azims=scene_info["cameras"]["azims"],
                                            elevs=scene_info["cameras"]["elevs"],
                                            device=device)
    rasterizer = create_rasterizer(cameras, img_size=img_size)
    shaders = {shader_type: create_shader(shader_type, cameras, lights, obj_source="3dfuture", device=device) for shader_type in shader_types}
    renderers = {shader_type: create_renderer(rasterizer, shaders[shader_type]) for shader_type in shader_types}

    batched_scenes = join_meshes_as_batch([scene] * num_views)
    scene_images = {image_type: renderers[image_type](batched_scenes,
                                                obj_meshes=meshes,
                                                blend_params=BlendParams(background_color=(0., 0., 0.) if image_type=="instances" else (1., 1., 1.)))
                    for image_type in shader_types}
    obj_images = [renderers["rgb"](join_meshes_as_batch([m.to(device)] * num_views)) for m in mesh_list]

    write_scene_images(scene_dir, scene_images, obj_images, num_views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/moos/scenes")
    parser.add_argument('--out_dir', type=str, default="./data/moos")
    parser.add_argument('--scene_name', type=str, default='scene_00000')
    args = parser.parse_args()
    
    reconstruct_one_scene(args.scene_name, args.data_dir, args.out_dir)