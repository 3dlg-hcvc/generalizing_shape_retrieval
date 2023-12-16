import warnings
warnings.filterwarnings('ignore')

import os
import json
import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rotation
from shapely.geometry import Polygon

# Util function for loading meshes
from pytorch3d.io import load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import (
    TexturesUV,
    TexturesVertex,
    TexturesAtlas,
)
    
torch.set_grad_enabled(False)


obj_path_shapenet = "./data/shapenet/ShapeNetCore.v2/{obj_id}/models/model_normalized.obj"
obj_path_3dfuture = "./data/3dfront/3D-FUTURE-model/{obj_id}/raw_model.obj"
obj_path_3dfuture_normalized = "./data/3dfront/3D-FUTURE-model/{obj_id}/normalized_model.obj"
obj_path_pix3d = "./data/pix3d/data/model/{obj_id}.obj"

problematic_raw_obj_ids = ['7e101ef3-7722-4af8-90d5-7c562834fabd'] # cause RuntimeError: CUDA error: device-side assert triggered during rendering


def load_3dfuture_metadata():
    obj_paths = json.load(open("./data/3dfuture/obj_paths_filtered.json"))
    obj_scales_issue = json.load(open("./data/3dfuture/obj_dims_issue.json"))

    obj_map = {cat: [] for cat in obj_paths}
    for cat in obj_paths:
        for p in obj_paths[cat]:
            obj_id = p.split('/')[-2]
            obj_map[cat].append(obj_id)

    return obj_map, obj_scales_issue


def load_3dfuture_obj(obj_id, 
                      load_textures=True,
                      obj_scales=None,
                      device="cpu",
                      **kwargs):
    if obj_id in problematic_raw_obj_ids:
        obj_filename = obj_path_3dfuture_normalized.format(obj_id=obj_id)
    else:
        obj_filename = obj_path_3dfuture.format(obj_id=obj_id)
    
    verts, faces, aux = load_obj(
        obj_filename,
        load_textures=load_textures,
        device=device
    )
    textures = None
    if load_textures:
        # TexturesUV type
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)
            image = list(tex_maps.values())[0].to(device)[None]
            textures = TexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image)
    
    if obj_scales is not None and obj_id in obj_scales and obj_id not in problematic_raw_obj_ids:
        verts /= 100.

    return verts, faces.verts_idx, textures


def load_shapenet_metadata():
    synset2cat = {'02818832': 'bed', '03001627': 'chair', '04256520': 'sofa', '04379243': 'table'}
    cat2synset = {v: k for k, v in synset2cat.items()}
    obj_scales = pd.read_csv("./data/shapenet/obj_scales.csv")
    obj_paths = [p.strip() for p in open("./data/shapenet/obj_paths.txt")]

    obj_map = {}
    for p in obj_paths:
        synset_id, obj_id = p.split('/')[1:3]
        cat = synset2cat[synset_id]
        full_obj_id = f"{synset_id}/{obj_id}"
        if cat not in obj_map:
            obj_map[cat] = [full_obj_id]
        else:
            obj_map[cat].append(full_obj_id)

    return obj_map, obj_scales


def load_shapenet_obj(full_obj_id, 
                      load_textures=True,
                      obj_scales=None,
                      use_pre_norm=False,
                      device="cpu",
                      **kwargs):
    obj_filename = obj_path_shapenet.format(obj_id=full_obj_id)
    synset_id, obj_id = full_obj_id.split('/')
    
    verts, faces, aux = load_obj(
        obj_filename,
        load_textures=load_textures,
        create_texture_atlas=load_textures,
        texture_atlas_size=4,
        device=device
    )
    textures = None
    if load_textures:
        textures = aux.texture_atlas
        # Some meshes don't have textures. In this case create a white texture map
        if textures is None:
            textures = verts.new_ones(faces.verts_idx.shape[0], 4, 4, 3)
        textures=TexturesAtlas(atlas=[textures])
        
    if use_pre_norm:
        assert obj_scales is not None
        dims = obj_scales[obj_scales.id == obj_id]["aligned.dims"].iloc[0]
        if dims is not np.nan:
            dims = [float(dim) for dim in dims.split('\,')]
            scale = torch.tensor(dims, device=device) / 100 / (verts.max(dim=0)[0] - verts.min(dim=0)[0])
            verts *= scale

    return verts, faces.verts_idx, textures


def load_pix3d_obj(full_obj_id, 
                    load_textures=True,
                    device="cpu",
                    **kwargs):
    obj_filename = obj_path_pix3d.format(obj_id=full_obj_id)
    
    verts, faces, aux = load_obj(
        obj_filename,
        load_textures=load_textures,
        device=device
    )
    textures = None
    if load_textures:
        textures = aux.texture_atlas
        # Some meshes don't have textures. In this case create a white texture map
        if textures is None:
            textures = verts.new_ones(faces.verts_idx.shape[0], 4, 4, 3)
        textures=TexturesAtlas(atlas=[textures])

    return verts, faces.verts_idx, textures


def random_sample_objects(obj_map, cat, obj_num=1, replace=True):
    random_indices = np.random.choice(len(obj_map[cat]), obj_num, replace=replace).tolist()
    object_ids = []
    for idx in random_indices:
        object_ids.append((cat, obj_map[cat][idx]))
    
    return object_ids


def random_transform_object(obj):
    verts = obj["verts"]
    x_min, y_min, z_min = verts.min(0)[0].cpu().numpy()
    x_max, y_max, z_max = verts.max(0)[0].cpu().numpy()
    short_side = min(x_max-x_min, z_max-z_min) / 2
    obj["short_side"] = short_side
    
    corners2d = [[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]]
    corners3d = [[x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_min, y_min, z_max],
                 [x_min, y_max, z_min], [x_max, y_max, z_min], [x_max, y_max, z_max], [x_min, y_max, z_max]]
    center3d = [(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2]
    
    rot = np.random.rand() * 360
    rot_mat2d = Rotation.from_euler('Z', rot, degrees=True).as_matrix()[:2, :2]
    obj["rot"] = rot
    obj["corners2d"] = (rot_mat2d @ np.array(corners2d).T).T
    
    rot_mat3d = Rotation.from_euler('XYZ', [0, -rot, 0], degrees=True).as_matrix().astype(np.float32)
    rot_mat3d = torch.tensor(rot_mat3d).type_as(verts)
    obj["rot_mat3d"] = rot_mat3d
    obj["verts"] = (rot_mat3d @ verts.T).T # + trans_mat
    obj["corners3d"] = (rot_mat3d @ np.array(corners3d).T).T
    obj["center3d"] = (rot_mat3d @ np.array(center3d).T).T
    
    
def layout_generation(objs, 
                      delta_distance=0.05, 
                      visualize2d=False,
                      out_dir=None,
                      visualize_layout_iterations=False):
    placed = []
    polygons_2d = []
    for i, obj in enumerate(objs):
        corners2d = obj["corners2d"]
        base_move = np.array([0., 0.])
        if placed == []:
            move = np.array([0., 0.])
        else:
            base_move = sum([o["move2d"] for o in placed]) / len(placed)
            move_dir = np.random.uniform(-1, 1, 2)
            move_dir /= np.linalg.norm(move_dir)
            move_dist = obj["short_side"] + sum([o["short_side"] for o in placed[:3]])
            move = move_dist * move_dir + base_move
        corners2d += move
        p = Polygon(corners2d)
        while check_intersections(p, polygons_2d):
            move_dist += delta_distance
            move = move_dist * move_dir + base_move
            corners2d += (move_dir * delta_distance)
            p = Polygon(corners2d)
        obj["move2d"] = move
        obj["corners2d"] = corners2d
        placed.append(obj)
        polygons_2d.append(p)
        
        if visualize_layout_iterations:
            assert out_dir is not None
            fig, ax = plt.subplots(figsize=(5,5))
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.invert_yaxis()
            for p in polygons_2d:
                ax.plot(*p.exterior.xy)
                fig.savefig(os.path.join(out_dir, f"layout2d_{i}.png"))
    
    scene_corners2d = np.concatenate([o["corners2d"] for o in placed])
    scene_dims2d = scene_corners2d.max(0) - scene_corners2d.min(0)
    scene_center2d = (scene_corners2d.max(0) + scene_corners2d.min(0)) / 2
    for obj in placed:
        obj["move2d"] -= scene_center2d
        obj["corners2d"] -= scene_center2d
        obj["polygon2d"] = Polygon(obj["corners2d"])
        trans3d = np.array([obj["move2d"][0], -torch.min(obj["verts"][:, 1]).item(), obj["move2d"][1]]) #.astype(np.float32)
        trans3d = torch.tensor(trans3d).type_as(obj["verts"])
        obj["trans3d"] = trans3d
        obj["verts"] += trans3d
        obj["center3d"] += trans3d #(obj["verts"].max(0)[0] + obj["verts"].min(0)[0]) / 2
        obj["corners3d"] += trans3d
    
    if visualize2d:
        assert out_dir is not None
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.invert_yaxis()
        for obj in objs:
            ax.plot(*obj["polygon2d"].exterior.xy)
            fig.savefig(os.path.join(out_dir, "layout2d.png"))
    
    return scene_center2d, scene_dims2d


def check_intersections(tgt_polygon, polygons):
    return any([tgt_polygon.intersects(p) for p in polygons]) if polygons != [] else False


def create_meshes(objs, device="cpu"):
    mesh_list = []
    for obj in objs:
        mesh = Meshes(
            verts=[obj["verts"]],
            faces=[obj["faces"]],
            textures=obj["textures"],
        )
        mesh_list.append(mesh)
        
    try:
        scene = join_meshes_as_scene(mesh_list, include_textures=True).to(device)
    except ValueError:
        with open('join_scene_error.txt', 'a') as f:
            print(', '.join([obj["id"] for obj in objs]), file=f)
    
    return scene, mesh_list