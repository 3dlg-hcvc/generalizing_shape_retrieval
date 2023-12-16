import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from gcmic.utils.lfd_util import extract_efd_feature, extract_zernike_moments_feature
from gcmic.utils.img_util import resize_padding, get_largest_contour


image_path = [p.strip() for p in open("./data/pix3d/pix3d_img_path.txt")]
img_map = {}
for p in image_path:
    name = p.split('.')[0][4:]
    img_map[name] = p

def prepare_pix3d_gt(img_name, pose_info, device, out_dir, save_vis=False):
    img_size = tuple(pose_info["img_size"])
    gt_mask_path = f"./data/pix3d/data/mask/{img_name}.png"
    gt_rgb_path = f"./data/pix3d/data/{img_map[img_name]}"
    m2f_mask_path = f"./data/pix3d/data/mask2former_swin-L_ade20k-semantic-all/{img_name}.sem_mask.png"
    
    gt_mask_np = np.asarray(Image.open(gt_mask_path).resize(img_size, Image.NEAREST))
    gt_mask_t = torch.from_numpy(gt_mask_np).to(device)
    m2f_mask_np = np.asarray(Image.open(m2f_mask_path).resize(img_size, Image.NEAREST))
    if m2f_mask_np.sum() > 20:
        m2f_mask_bool = m2f_mask_np.astype(bool)
    else:
        m2f_mask_bool = np.ones_like(m2f_mask_np).astype(bool)
    occlusion_mask = m2f_mask_bool.copy()
    bbox = cv2.boundingRect((m2f_mask_bool * 255).astype("ubyte"))
    x, y, w, h = bbox
    
    if save_vis:
        os.makedirs(os.path.join(out_dir, pose_info["img_name"]), exist_ok=True)
        m2f_mask = np.asarray(Image.open(m2f_mask_path).resize((512, 512), Image.NEAREST)).astype(bool)
        gt_rgb_np = np.asarray(Image.open(gt_rgb_path).convert('RGB').resize((512, 512)))
        gt_seg = gt_rgb_np.copy()
        gt_seg[m2f_mask] = [53, 155, 219]
        rgb_seg = (gt_seg * 0.3 + gt_rgb_np * (1 - 0.3)).astype(np.uint8)
        Image.fromarray(rgb_seg).save(os.path.join(out_dir, pose_info["img_name"], "img_query.png"))
        rgb_seg = np.asarray(Image.fromarray(rgb_seg).resize(img_size))
        resize_padding(Image.fromarray(rgb_seg[y:y + h, x:x + w, :]), 224, mode="RGB", background_color=(255, 255, 255)).save(os.path.join(out_dir, pose_info["img_name"], "obj_crop.png"))
        resize_padding(Image.fromarray(m2f_mask_bool[y:y + h, x:x + w]), 224, mode="L").save(os.path.join(out_dir, pose_info["img_name"], "mask_crop.png"))
    
    gt_rgb_np = np.asarray(Image.open(gt_rgb_path).convert('RGB').resize(img_size))
    gt_rgb_np = gt_rgb_np * np.expand_dims(m2f_mask_bool, 2)
    gt_rgb_np[~m2f_mask_bool] = [255, 255, 255]
    gt_rgb_crop = gt_rgb_np[y:y + h, x:x + w, :]
    gt_rgb_crop_im = resize_padding(Image.fromarray(gt_rgb_crop), 100, mode="RGB", background_color=(255, 255, 255))
    gt_rgb_crop = TF.to_tensor(gt_rgb_crop_im).to(device)
    
    gt_mask_lfd = None
    if gt_mask_np.sum() > 20:
        gt_fourier = extract_efd_feature(get_largest_contour(gt_mask_np))
        gt_moments = extract_zernike_moments_feature(gt_mask_np, radius=128, degree=10)
        gt_mask_lfd = np.concatenate([gt_fourier, gt_moments])
    
    return gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd


def prepare_scan2cad_gt(pose_info, device, out_dir, save_vis=False):
    mask_name = pose_info["mask_name"]
    img_name = pose_info["img_name"]
    scene_id, img_id = img_name.split('/')
    img_size = tuple(pose_info["img_size"])
    gt_mask_path = f"./data/scan2cad/mask/{mask_name}.png"
    gt_rgb_path = f"./data/scan2cad/Images/tasks/scannet_frames_25k/{scene_id}/color/{img_id}.jpg"
    
    gt_mask_np = np.asarray(Image.open(gt_mask_path).resize(img_size, Image.NEAREST)).astype(np.uint8)
    gt_mask_bool = gt_mask_np.astype(bool)
    gt_mask_t = torch.from_numpy(gt_mask_np).to(device)
    occlusion_mask = gt_mask_bool.copy()
    bbox = cv2.boundingRect((gt_mask_bool * 255).astype("ubyte"))
    x, y, w, h = bbox
    
    if save_vis:
        os.makedirs(os.path.join(out_dir, mask_name), exist_ok=True)
        gt_rgb_np = np.asarray(Image.open(gt_rgb_path).convert('RGB').resize(img_size))
        gt_seg = gt_rgb_np.copy()
        gt_seg[gt_mask_bool] = [53, 155, 219]
        rgb_seg = (gt_seg * 0.3 + gt_rgb_np * (1 - 0.3)).astype(np.uint8)
        Image.fromarray(rgb_seg).save(os.path.join(out_dir, mask_name, "obj_query.png"))
    
    gt_rgb_np = np.asarray(Image.open(gt_rgb_path).convert('RGB').resize(img_size))
    gt_rgb_np = gt_rgb_np * np.expand_dims(gt_mask_bool, 2)
    gt_rgb_np[~gt_mask_bool] = [255, 255, 255]
    gt_rgb_crop = gt_rgb_np[y:y + h, x:x + w, :]
    gt_rgb_crop_im = resize_padding(Image.fromarray(gt_rgb_crop), 100, mode="RGB", background_color=(255, 255, 255))
    gt_rgb_crop = TF.to_tensor(gt_rgb_crop_im).to(device)
    
    gt_mask_lfd = None
    if gt_mask_np.sum() > 20:
        gt_fourier = extract_efd_feature(get_largest_contour(gt_mask_np))
        gt_moments = extract_zernike_moments_feature(gt_mask_np, radius=128, degree=10)
        gt_mask_lfd = np.concatenate([gt_fourier, gt_moments])
    
    return gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd


def prepare_moos_gt(pose_info, device, out_dir, save_vis=False):
    img_size = tuple(pose_info["img_size"])
    scene_id, obj_scene_id, view_id = pose_info["full_view_id"].split('_')
    gt_mask_path = f"./data/moos/scenes/scene_{int(scene_id):05d}/objects/{obj_scene_id}_{view_id}.mask.png"
    gt_insts_path = f"./data/moos/scenes/scene_{int(scene_id):05d}/instances/instances_{view_id}.png"
    gt_rgb_path = f"./data/moos/scenes/scene_{int(scene_id):05d}/rgb/rgb_{view_id}.png"
    
    insts = np.asarray(Image.open(gt_insts_path).convert("L"))
    gt_rgb_np = np.asarray(Image.open(gt_rgb_path).convert('RGB'))
    insts = np.rint(insts / 255 * 4)
    gt_inst_mask = (insts == (int(obj_scene_id)+1))
    occlusion_mask = np.isin(insts, [0, int(obj_scene_id)+1])
    bbox = cv2.boundingRect((gt_inst_mask * 255).astype("ubyte"))
    x, y, w, h = bbox
    
    if save_vis:
        os.makedirs(os.path.join(out_dir, pose_info["full_view_id"]), exist_ok=True)
        gt_seg = gt_rgb_np.copy()
        gt_seg[gt_inst_mask] = [235, 147, 54]
        rgb_seg = (gt_seg * 0.3 + gt_rgb_np * (1 - 0.3)).astype(np.uint8)
        resize_padding(Image.fromarray(rgb_seg[y:y + h, x:x + w, :]), 224, mode="RGB", background_color=(255, 255, 255)).save(os.path.join(out_dir, pose_info["full_view_id"], "obj_crop.png"))
        gt_mask_crop = gt_inst_mask[y:y + h, x:x + w]
        resize_padding(Image.fromarray(gt_mask_crop), 224, mode="L").save(os.path.join(out_dir, pose_info["full_view_id"], "mask_crop.png"))
    
    gt_rgb_np = gt_rgb_np * np.expand_dims(gt_inst_mask, 2)
    gt_rgb_np[~gt_inst_mask] = [255, 255, 255]
    gt_rgb_crop = gt_rgb_np[y:y + h, x:x + w, :]
    gt_rgb_crop_im = resize_padding(Image.fromarray(gt_rgb_crop), 100, mode="RGB", background_color=(255, 255, 255))
    gt_rgb_crop = TF.to_tensor(gt_rgb_crop_im).to(device)
    
    gt_mask = Image.open(gt_mask_path).resize(img_size, Image.NEAREST)
    gt_mask_np = np.array(gt_mask).astype(np.uint8)
    gt_mask_t = torch.from_numpy(gt_mask_np).to(device)
    gt_mask_lfd = None
    if gt_mask_np.sum() > 20:
        gt_fourier = extract_efd_feature(get_largest_contour(gt_mask_np))
        gt_moments = extract_zernike_moments_feature(gt_mask_np, radius=128, degree=10)
        gt_mask_lfd = np.concatenate([gt_fourier, gt_moments])
    
    return gt_rgb_crop, gt_mask_t, occlusion_mask, bbox, gt_mask_lfd