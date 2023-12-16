import sys
import argparse
from tqdm import tqdm
import h5py
import numpy as np
from PIL import Image

sys.path.append("./")
from gcmic.utils.lfd_util import *
from gcmic.utils.img_util import *


def get_obj_lfd(obj_id, num_views=200):
    lfds = []
    for view_idx in range(num_views):
        mask_path = mask_path_template.format(obj_id=obj_id, view_idx=view_idx)
        if obj_id in missing_mv_obj_ids:
            mask_path = mask_path.replace('normalized', 'raw')
        mask = Image.open(mask_path).resize((256, 256), Image.NEAREST)
        mask = np.array(mask)
        
        # extract contour
        cnt = get_largest_contour(mask)  
        # compute Fourier and Zernike moments to characterize the shape
        fourier_desc = extract_efd_feature(cnt)
        moments_desc = extract_zernike_moments_feature(mask, radius=128, degree=10)
        lfd = np.concatenate([fourier_desc, moments_desc])
        lfds.append(lfd)
    lfds = np.concatenate([lfds], axis=0)
    return lfds


if __name__ == '__main__':
    data_dir = "/path_to_3dfuture"
    mask_dir = f"{data_dir}/lfd_200"
    mask_path_template = data_dir + "/lfd_200/{obj_id}/normalized_model-{view_idx}.color.encoded.png"
    obj_ids = [n.strip() for n in open("./data/3dfuture/obj_ids.txt")]
    missing_mv_obj_ids = [n.strip() for n in open("./data/3dfuture/missing_mv_obj_ids.txt")]
    num_obj = len(obj_ids)

    with h5py.File(f"{data_dir}/lfd_200.h5", "w") as f:
        f.create_dataset(
            'obj_id',
            shape=(num_obj, ),
            dtype=h5py.string_dtype(encoding='utf-8'),
            data=obj_ids
        )
        lfd_dataset = f.create_dataset(
            'lfd',
            shape=(num_obj, 200, 45),
            dtype=np.float32,
            chunks=(1, 200, 45),
            compression='gzip',
            compression_opts=9,
            shuffle = True,
        )
        for idx in tqdm(range(num_obj)):
            obj_id = obj_ids[idx]
            lfds = get_obj_lfd(obj_id)
            lfd_dataset[idx] = lfds