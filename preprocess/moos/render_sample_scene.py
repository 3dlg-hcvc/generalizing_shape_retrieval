import warnings
warnings.filterwarnings('ignore')

import os
import json
import random
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.renderer import BlendParams

from obj_util import (
    load_3dfuture_metadata, 
    load_3dfuture_obj, 
    load_shapenet_metadata, 
    load_shapenet_obj,
    random_sample_objects,
    random_transform_object,
    layout_generation,
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
img_size = (1024, 1024)
shader_types = ["rgb", "instances", "depth", "normal"]


@torch.no_grad()
def create_one_scene(out_dir, 
                     scene_id,
                     obj_ids,
                     obj_scales,
                     load_obj_func,
                     obj_source="3dfuture"):
    random.shuffle(obj_ids)
    
    objs = []
    for cat, obj_id in obj_ids:
        obj = {"id": obj_id, 
               "category": cat,
               "source": obj_source}
        obj["verts"], obj["faces"], obj["textures"] = load_obj_func(obj_id, obj_scales=obj_scales)
        random_transform_object(obj)
        objs.append(obj)
    scene_center2d, scene_dims2d = layout_generation(objs, visualize2d=True, out_dir=out_dir)
    scene_center3d = sum([obj["center3d"] for obj in objs]) / len(objs)
    scene_height = max([obj["center3d"][1].item() for obj in objs])
    
    scene, mesh_list = create_meshes(objs, device=device)
    meshes = join_meshes_as_batch(mesh_list).to(device)
    
    lights = create_ligths(location=((0., scene_height*2+0.5, 0.),), device=device)
    camera_dist = max(scene_dims2d) + 0.5,
    camera_look_at = (0., scene_center3d[1].item(), 0.)
    cameras, azims, elevs = create_cameras(dist=camera_dist,
                                            at=(camera_look_at,),
                                            img_size=img_size,
                                            azim_step=30,
                                            elev_step=10,
                                            elev_levels=2,
                                            elev_start=5,
                                            elev_end=25,
                                            elev_rand=True,
                                            device=device)
    num_views = len(azims)
    rasterizer = create_rasterizer(cameras, img_size=img_size)
    shaders = {shader_type: create_shader(shader_type, cameras, lights, obj_source="3dfuture", device=device) for shader_type in shader_types}
    renderers = {shader_type: create_renderer(rasterizer, shaders[shader_type]) for shader_type in shader_types}
    
    batched_scenes = join_meshes_as_batch([scene] * num_views)
    scene_images = {image_type: renderers[image_type](batched_scenes,
                                                obj_meshes=meshes,
                                                blend_params=BlendParams(background_color=(0., 0., 0.) if image_type=="instances" else (1., 1., 1.)))
                    for image_type in shader_types}
    obj_images = [renderers["rgb"](join_meshes_as_batch([m.to(device)] * num_views)) for m in mesh_list]
    
    write_scene_images(out_dir, scene_images, obj_images, num_views)
    
    scene_info = {
        "id": scene_id,
        "objects": [
            {
                "id": obj["id"],
                "category": obj["category"],
                "source": obj["source"],
                "textures": True,
                "rot": obj["rot"],
                "rot_mat3d": obj["rot_mat3d"].cpu().numpy().tolist(),
                "trans3d": obj["trans3d"].cpu().numpy().tolist(),
                "center3d": obj["center3d"].tolist(),
                "corners3d": obj["corners3d"].tolist(), 
            }
            for obj in objs
        ],
        "cameras": {
            "dist": camera_dist,
            "look_at": camera_look_at,
            "focal_length": 35,
            "sensor_width": 32,
            "azim_step": 30,
            "elev_step": 10,
            "elev_levels": 2,
            "elev_start": 5,
            "elev_end": 25,
            "azims": azims,
            "elevs": elevs,
            "elev_rand": True,
            "positions": cameras.get_camera_center().cpu().numpy().tolist()
        },
        "lights": {
            "type": "point_light",
            "positions": lights.location.cpu().numpy().tolist()
        },
        "metadata": {
            "img_size": img_size,
            "num_views": num_views,
            "render_types": shader_types
        }
    }
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(scene_info, f)
    
# create_one_scene("./preprocess/moos/example")


def create_scenes(out_dir, num_scenes=1, obj_source="3dfuture"):
    if obj_source == "shapenet":
        obj_map, obj_scales = load_shapenet_metadata()
        load_obj = load_shapenet_obj
    elif obj_source == "3dfuture":
        obj_map, obj_scales = load_3dfuture_metadata()
        load_obj = load_3dfuture_obj
    
    all_obj_ids = {cat: [] for cat in cats}
    for cat in cats:
        all_obj_ids[cat].extend(random_sample_objects(obj_map, cat, num_scenes))
    
    for id in tqdm(range(num_scenes)):
        torch.cuda.empty_cache()
        
        scene_name = f"scene_{id:05d}"
        scene_dir = os.path.join(out_dir, scene_name)
        os.makedirs(os.path.join(scene_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "instances"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "normal"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "objects"), exist_ok=True)
        
        obj_ids = [all_obj_ids[cat][id] for cat in cats]
    
        create_one_scene(scene_dir, id, obj_ids, obj_scales, load_obj)


def create_one_scene_worker(out_dir, scene_id, obj_source="3dfuture"):
    if obj_source == "shapenet":
        obj_map, obj_scales = load_shapenet_metadata()
        load_obj = load_shapenet_obj
    elif obj_source == "3dfuture":
        obj_map, obj_scales = load_3dfuture_metadata()
        load_obj = load_3dfuture_obj
    
    obj_ids = []
    for cat in cats:
        obj_ids.extend(random_sample_objects(obj_map, cat, cat2num[cat]))
    
    torch.cuda.empty_cache()
    
    scene_name = f"scene_{scene_id:05d}"
    scene_dir = os.path.join(out_dir, scene_name)
    os.makedirs(os.path.join(scene_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "instances"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(scene_dir, "objects"), exist_ok=True)
    
    create_one_scene(scene_dir, scene_id, obj_ids, obj_scales, load_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='./data/moos')
    parser.add_argument('--scene_id', type=int, default=0)
    parser.add_argument('--obj_source', type=str, default='3dfuture')
    args = parser.parse_args()
    
    create_one_scene_worker(args.out_dir, args.scene_id, args.obj_source)
    