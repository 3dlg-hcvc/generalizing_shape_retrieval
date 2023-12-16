import os, sys, argparse
import json
import h5py
from tqdm import tqdm
from omegaconf import OmegaConf
from glob import glob
import numpy as np
import pandas as pd
from PIL import Image
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, sample_farthest_points

sys.path.append("./")
from gcmic.utils.io_util import load_obj, normalize_verts


def gen_image_h5(output_path, cfg, img_size=(224, 224)):
    data_dir = cfg.DATA_PATH.moos.raw
    
    annotation = pd.read_csv("./data/moos/moos_annotation.txt")
    img_names = annotation['scene_img_path'].str.split('.').str[0].unique()
    full_view_ids = annotation['full_view_id']
    
    H, W = img_size

    with h5py.File(output_path, 'a') as f:
        f.create_dataset(
            "full_view_id",
            shape=(len(annotation),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=full_view_ids.tolist()
        )
        f.create_dataset(
            "img_names",
            shape=(len(img_names),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=img_names.tolist()
        )
        # create dataset and save image
        img_dataset = f.create_dataset(
            "image",
            shape=(len(img_names), H, W, 3),
            dtype=np.uint8,
            chunks=(1, H, W, 3),
            compression="gzip",
            compression_opts=9,
        )
        mask_dataset = f.create_dataset(
            "mask",
            shape=(len(img_names), H, W),
            dtype=np.uint8,
            chunks=(1, H, W),
            compression="gzip",
            compression_opts=9,
        )
        # populate dataset
        for idx, img_name in enumerate(tqdm(img_names)):
            img_path = os.path.join(data_dir, f"{img_name}.png")
            mask_path = os.path.join(data_dir, f"{img_name.replace('rgb', 'instances')}.png")
            img = np.asarray(Image.open(img_path).convert('RGB').resize(img_size))
            mask = np.asarray(Image.open(mask_path).convert("L").resize(img_size, Image.NEAREST))
            img_dataset[idx] = img
            mask_dataset[idx] = mask


def gen_object_h5(output_path, cfg, num_points=2048, use_normalized=True, sampling="face_area"):
    data_dir = os.path.join(cfg.DATA_PATH.future3d.raw, "3D-FUTURE-model")
    obj_ids = [obj_id.strip() for obj_id in open("./data/3dfuture/obj_ids.txt")]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            "obj_ids",
            shape=(len(obj_ids),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_ids
        )
        points_dataset = f.create_dataset(
            "obj_points",
            shape=(len(obj_ids), num_points, 3),
            dtype=np.float32,
            chunks=(1, num_points, 3),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        normals_dataset = f.create_dataset(
            "obj_normals",
            shape=(len(obj_ids), num_points, 3),
            dtype=np.float32,
            chunks=(1, num_points, 3),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        # populate 3dfuture objects 
        for idx, obj_id in enumerate(tqdm(obj_ids)):
            obj_path = os.path.join(data_dir, obj_id, 'raw_model.obj') # some .obj files use wrong .mtl paths, but it's fine when load_textures=False
            if use_normalized:
                obj_path = obj_path.replace('raw', 'normalized')
            mesh = load_obj(obj_path)
            verts, faces = (
                    mesh[0].clone(),
                    mesh[1].clone(),
                )
            if use_normalized:
                verts = normalize_verts(verts, scale_along_diagonal=True)
            device = "cuda:0"
            verts = verts.to(device)
            faces = faces.to(device)
            p3d_mesh = Meshes(verts=[verts], faces=[faces]).to(device)
            if sampling == "face_area":
                points, normals = sample_points_from_meshes(p3d_mesh, num_points, return_normals=True)
            elif sampling == "farthest":
                over_points, over_normals = sample_points_from_meshes(p3d_mesh, 1000000, return_normals=True)
                points, indices = sample_farthest_points(over_points, K=num_points)
                normals = over_normals[:, indices.squeeze(), :]
            else:
                raise "Smapling method not supported"
            points_dataset[idx] = np.asarray(points.squeeze().cpu())
            normals_dataset[idx] = np.asarray(normals.squeeze().cpu())


def gen_multiview_h5(output_path, cfg, img_size=(224, 224)):
    data_dir = os.path.join(cfg.DATA_PATH.future3d.preprocessed, cfg.data.multiview.mv_dirname)
    
    obj_ids = sorted(os.listdir(data_dir))
    models = json.load(open(os.path.join(cfg.DATA_PATH.future3d.raw, "3D-FUTURE-model/model_info.json")))
    obj2cat = {}
    for m in models:
        obj2cat[m["model_id"]] = m["super-category"].lower()
    obj_cats = [obj2cat[obj_id] for obj_id in obj_ids]
    
    H, W = img_size
    
    with h5py.File(output_path, 'a') as f:
        f.create_dataset(
            "obj_ids",
            shape=(len(obj_ids),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_ids
        )
        f.create_dataset(
            "cats",
            shape=(len(obj_ids),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_cats
        )
        # create dataset to save multiviews
        h5_dataset = f.create_dataset(
            "multiview",
            shape=(len(obj_ids), 12, H, W),
            dtype=np.uint8,
            chunks=(1, 12, H, W),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        # populate dataset
        for idx, obj_id in enumerate(tqdm(obj_ids)):
            multiviews = sorted(glob(os.path.join(data_dir, obj_id, '*.png')))
            mv_set = []
            for mv_path in multiviews:
                mv = Image.open(mv_path).convert('L').resize(img_size)
                mv = np.asarray(mv)
                mv = np.expand_dims(mv, axis=0)
                mv_set.append(mv)
            mv_set = np.concatenate(mv_set, axis=0)
            h5_dataset[idx] = mv_set
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', help='The output HDF5 path')
    parser.add_argument('--conf', default='conf/dataset/moos.yaml', help='The input HDF5 path')
    args = parser.parse_args()
    conf_list = ['conf/default.yaml', args.conf]
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in conf_list])
    
    gen_image_h5("./data/moos/moos_1k.h5", cfg, (1024, 1024))
    gen_multiview_h5("./data/moos/moos_mv.h5", cfg)
    gen_object_h5("./data/moos/moos_obj.h5", cfg, 4096, sampling="farthest")
    
