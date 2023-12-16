import numpy as np
import torch
import pytorch3d.io as p3d_io


def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = verts.clone().t()
    if R is not None:
        assert R.dim() == 2
        assert R.size(0) == 3 and R.size(1) == 3
        rot_verts = torch.mm(R, rot_verts)
    if t is not None:
        assert t.dim() == 1
        assert t.size(0) == 3
        rot_verts = rot_verts + t.unsqueeze(1)
    rot_verts = rot_verts.t()
    return rot_verts


def normalize_verts(verts, scale_along_diagonal=False):
    # centering and normalization
    min_vert, _ = torch.min(verts, 0)
    min_x, min_y, min_z = min_vert
    max_vert, _ = torch.max(verts, 0)
    max_x, max_y, max_z = max_vert
    x_ctr = (min_x + max_x) / 2.0
    y_ctr = (min_y + max_y) / 2.0
    z_ctr = (min_z + max_z) / 2.0
    if scale_along_diagonal:
        diag_scale = 1 / np.linalg.norm([max_x - min_x, max_y - min_y, max_z - min_z])
        x_scale = y_scale = z_scale = diag_scale
    else:
        x_scale = 1.0 / (max_x - min_x)
        y_scale = 1.0 / (max_y - min_y)
        z_scale = 1.0 / (max_z - min_z)
    verts[:, 0] = (verts[:, 0] - x_ctr) * x_scale
    verts[:, 1] = (verts[:, 1] - y_ctr) * y_scale
    verts[:, 2] = (verts[:, 2] - z_ctr) * z_scale
    return verts


def load_ply(pcd_path, normalize=True):
    pcd = p3d_io.load_ply(pcd_path)
    if normalize:
        return normalize_verts(pcd[0])
    else:
        return pcd[0]

    
def load_obj(obj_path):
    mesh = p3d_io.load_obj(obj_path, load_textures=False)
    return [mesh[0], mesh[1].verts_idx]