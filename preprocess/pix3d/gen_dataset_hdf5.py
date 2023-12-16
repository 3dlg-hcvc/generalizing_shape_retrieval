import os, sys, argparse
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


def gen_image_h5(output_path, cfg):
    data_dir = cfg.DATA_PATH.pix3d.raw
    prep_dir = cfg.DATA_PATH.pix3d.preprocessed
    m2f_dir = "/local-scratch/qiruiw/dataset/pix3d/mask2former_swin-L_ade20k-semantic-all"
    
    annotation = pd.read_csv(os.path.join(prep_dir, cfg.data.annotation_file))
    img_names = annotation['image_path'].str.replace('img/', '').str.split('.').str[0]

    with h5py.File(output_path, 'a') as f:
        f.create_dataset(
            "names",
            shape=(len(annotation),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=img_names.tolist()
        )
        # create dataset and save image
        img_dataset = f.create_dataset(
            "image",
            shape=(len(annotation), 224, 224, 3),
            dtype=np.uint8,
            chunks=(1, 224, 224, 3),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        mask_dataset = f.create_dataset(
            "mask",
            shape=(len(annotation), 224, 224),
            dtype=bool,
            chunks=(1, 224, 224),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        m2f_dataset = f.create_dataset(
            "m2f_mask",
            shape=(len(annotation), 224, 224),
            dtype=bool,
            chunks=(1, 224, 224),
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )
        # populate dataset
        for idx in tqdm(range(len(annotation))):
            img_name = img_names[idx]
            img_path = os.path.join(data_dir, annotation.iloc[idx]['image_path'])
            mask_path = os.path.join(data_dir, annotation.iloc[idx]['mask_path'])
            m2f_path = os.path.join(m2f_dir, f"{img_name}.sem_mask.png")
            img = np.asarray(Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR))
            img_dataset[idx] = img
            mask = np.asarray(Image.open(mask_path).convert("L").resize((224, 224), Image.NEAREST))
            mask_dataset[idx] = mask
            m2f_mask = np.asarray(Image.open(m2f_path).convert("L").resize((224, 224), Image.NEAREST))
            m2f_dataset[idx] = m2f_mask


def gen_object_h5(output_path, cfg, num_points=2048, use_normalized=True, sampling="face_area"):
    data_dir = cfg.DATA_PATH.pix3d.raw
    
    obj_names = [obj.strip() for obj in open("./data/pix3d/pix3d_obj_names.txt")]

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            "obj_names",
            shape=(len(obj_names),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_names
        )
        points_dataset = f.create_dataset(
            "obj_points",
            shape=(len(obj_names), num_points, 3),
            dtype=np.float32,
            chunks=(1, num_points, 3),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        normals_dataset = f.create_dataset(
            "obj_normals",
            shape=(len(obj_names), num_points, 3),
            dtype=np.float32,
            chunks=(1, num_points, 3),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        # populate dataset
        for idx, obj_name in enumerate(tqdm(obj_names)):
            obj_dir = os.path.join(data_dir, 'model', obj_name)
            obj_path = sorted(glob(os.path.join(obj_dir, '*.obj')))[0]
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
    data_dir = os.path.join(cfg.DATA_PATH.pix3d.preprocessed, cfg.data.multiview.mv_dirname)
    
    obj_names = [obj.strip() for obj in open("./data/pix3d/pix3d_obj_names.txt")]
    H, W = img_size
    
    with h5py.File(output_path, 'w') as f:
        # create dataset to store obj_paths    
        f.create_dataset(
            "obj_names",
            shape=(len(obj_names),),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_names
        )
        # create dataset to save multiviews
        h5_dataset = f.create_dataset(
            "multiview",
            shape=(len(obj_names), 12, H, W),
            dtype=np.uint8,
            chunks=(1, 12, H, W),
            compression='gzip',
            compression_opts=9,
            shuffle=True,
        )
        # populate dataset
        for idx, obj_name in enumerate(tqdm(obj_names)):
            mv_dir = os.path.join(data_dir, obj_name)
            multiviews = sorted(glob(os.path.join(mv_dir, '**/*.png'), recursive=True))
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
    parser.add_argument('--conf', default='conf/dataset/pix3d.yaml', help='The input HDF5 path')
    args = parser.parse_args()
    conf_list = ['conf/default.yaml', args.conf]
    cfg = OmegaConf.merge(*[OmegaConf.load(f) for f in conf_list])
    
    gen_image_h5("./data/pix3d/pix3d_224.h5", cfg)
    gen_multiview_h5("./data/pix3d/pix3d_mv.h5", cfg)
    gen_object_h5("./data/pix3d/pix3d_obj.h5", cfg, 4096, sampling="farthest")
