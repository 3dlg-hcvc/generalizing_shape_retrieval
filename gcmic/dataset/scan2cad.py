import os, sys, json
import h5py

import random
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

from gcmic.dataset.sampler import UniqueRandomSampler
from gcmic.utils.img_util import resize_padding

CLASS_TO_IDX = {"chair": 0, "bed": 1, "sofa": 2,"table": 3}
IDX_TO_CLASS = {v:k for k,v in CLASS_TO_IDX.items()}

render_examples = ['scene0663_01/000100.3.png', 'scene0616_00/000600.1.png', 'scene0700_01/000900.3.png', 'scene0616_00/001700.6.png', 'scene0690_01/000300.2.png', 'scene0435_02/000400.1.png', 'scene0697_02/001400.1.png', 'scene0025_01/000700.2.png', 'scene0334_00/000000.3.png']


class Scan2CAD(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.task = cfg.general.task
        self.split = split
        
        self.data_dir = cfg.data.preprocessed_path
        self.h5_path = cfg.data.h5_path
        self.mv_path = cfg.data.mv_path
        self.pose_path = cfg.data.pose_path
        
        self.input_dim = cfg.data.input_dim
                
        self.img_source = cfg.data.img_source
        self.mask_source = cfg.data.mask_source
        self.use_crop = cfg.data.use_crop

        self.view_transform = self.get_multiview_transform()
            
        self._load_annotation()

    def _load_annotation(self):
        df = pd.read_csv(os.path.join(self.data_dir, self.cfg.data.annotation_file))
        
        if self.split != 'all':
            df = df[df.split == self.split]
        if self.cfg.data.center_in_image:
            df = df[df.center_in_image]
        # df = df[df.mask_path.isin(render_examples)]
        self.annotation = df
        
        with h5py.File(self.h5_path, 'r') as h5:
            self.img_names = [name.decode("utf8") for name in h5['img_names'][:]]
            self.mask_names = [name.decode("utf8") for name in h5['mask_names'][:]]
        with h5py.File(self.mv_path, 'r') as h5:
            self.obj_names = [name.decode("utf8") for name in h5['obj_names'][:]]
        self.pose_dict = json.load(open(self.pose_path))

    def __len__(self):
        return len(self.annotation)
    
    def sync_transform(self, img, mask):
        if self.split == "train":
            # RandomAffine
            affine_params = transforms.RandomAffine.get_params(degrees=[0.0, 0.0], translate=(0.05, 0.05), scale_ranges=None, shears=None, img_size=TF.get_image_size(img))
            img = TF.affine(img, *affine_params, interpolation=0, fill=0)
            mask = TF.affine(mask, *affine_params, interpolation=0, fill=0)
            
            # RandomResizedCrop
            resized_crop_params = transforms.RandomResizedCrop.get_params(img, scale=(0.85, 0.95), ratio=(3./4., 4./3.))
            img = TF.resized_crop(img, *resized_crop_params, size=(self.input_dim, self.input_dim))
            mask = TF.resized_crop(mask, *resized_crop_params, size=(self.input_dim, self.input_dim), interpolation=Image.NEAREST)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # Random Color Jittor
            if not self.cfg.data.use_color_transfer and random.random() > 0.5:
                img = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)(img)

        # To tensor
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        return img, mask
    
    def get_multiview_transform(self):
        transform_list = []
        if self.split == "train":
            transform_list.append(transforms.RandomResizedCrop(size=(self.input_dim, self.input_dim), scale=(0.65, 0.9)))
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def __getitem__(self, idx):
        cat = self.annotation.iloc[idx]['cat_id']
        cat_idx = CLASS_TO_IDX[cat]
        obj_name = '0' + str(self.annotation.iloc[idx]['sn_cat_id']) + '/' + self.annotation.iloc[idx]['sn_cad_id']
        img_name = self.annotation.iloc[idx]['image_name']
        mask_name = self.annotation.iloc[idx]['mask_path'][:-4]
        pose_info = self.pose_dict[mask_name]

        with h5py.File(self.h5_path, 'r') as h5:
            h5_img_idx = self.img_names.index(img_name)
            h5_mask_idx = self.mask_names.index(mask_name.replace('.', '_'))
            image = h5[self.img_source][h5_img_idx]
            mask = h5[self.mask_source][h5_mask_idx]
                
            if self.use_crop and mask.sum() != 0:
                mask = (mask * 255).astype("ubyte")
                x, y, w, h = cv2.boundingRect(mask)
                mask = mask[y:y + h, x:x + w]
                image = image[y:y + h, x:x + w, :]
                mask = resize_padding(Image.fromarray(mask), 224, mode="L")
                image = resize_padding(Image.fromarray(image), 224, mode="RGB", background_color=(255, 255, 255))
            else:
                mask = Image.fromarray(mask)
                image = Image.fromarray(image)
            
            image, mask = self.sync_transform(image, mask)

        with h5py.File(self.mv_path, 'r') as h5:
            obj_idx = self.obj_names.index(obj_name)
            multiview = h5['multiview'][obj_idx] # (12, 224, 224)
            multiview = torch.cat([self.view_transform(Image.fromarray(multiview[mv_id])).unsqueeze(0) for mv_id in range(len(multiview))], 0)
        
        data_dict = {"data_idx": idx, "cat_idx": cat_idx, "img_idx": idx, "img_name": img_name, "shape_idx": obj_idx, "shape": multiview, "shape_id": obj_name, "mask_name": mask_name, "pose_info": pose_info}
        data_dict["image"] = image
        data_dict["mask"] = mask
            
        return data_dict


def scan2cad_loader(cfg):
    
    def collate_fn(batch):
        data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                data[key] = torch.stack([sample[key] for sample in batch], axis=0)
            elif isinstance(batch[0][key], str):
                data[key] = [sample[key] for sample in batch]
            elif isinstance(batch[0][key], dict):
                data[key] = [sample[key] for sample in batch]
            else:
                data[key] = torch.tensor([sample[key] for sample in batch])
        return data

    if cfg.general.task in ['train', 'finetune']:
        splits = ['train', 'val']
    else:
        splits = ['val']
    
    datasets = {split: Scan2CAD(cfg, split) for split in splits}
    dataloaders = {
        split: DataLoader(datasets[split],
                    batch_size=cfg.data.batch_size,
                    shuffle=True if not cfg.data.unique_data_sampler and split == 'train' else False,
                    sampler=UniqueRandomSampler(datasets[split], labels=datasets[split].obj_names, batch_size=cfg.data.batch_size) if cfg.data.unique_data_sampler and split == 'train' else None, 
                    num_workers=cfg.data.num_workers,
                    drop_last=True if cfg.general.task in ['train', 'finetune'] else False,
                    pin_memory=True,
                    collate_fn=collate_fn if not cfg.data.unique_data_sampler else None)
        for split in splits
    }
    return datasets, dataloaders

