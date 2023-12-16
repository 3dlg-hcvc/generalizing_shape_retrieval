import warnings
warnings.filterwarnings('ignore')

import os
import itertools
import torch
import torch.nn.functional as F

import numpy as np
import imageio
from PIL import Image
from scipy.spatial.transform import Rotation as Rotation

# functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftPhongShader,
    HardPhongShader,
    HardFlatShader,
    SoftSilhouetteShader,
)
from pytorch3d.renderer.mesh.shader import ShaderBase, HardDepthShader
from pytorch3d.renderer.blending import hard_rgb_blend


def create_cameras(dist=5,
                   at=((0., 0., 0.),), 
                   img_size=(1024, 1024),
                   sensor_width=32,
                   azim_step=30,
                   azims=None,
                   elev_step=10,
                   elev_levels=3,
                   elev_start=10,
                   elev_end=30,
                   elev_rand=False,
                   elevs=None,
                   device="cpu"):
    w, h = img_size
    focal_length = 35 * w / sensor_width
    # R, T = look_at_view_transform(5, 15, 0, at=at)
    if (not azims) or (not elevs):
        azim = np.arange(0, 360, azim_step)
        if not elev_rand:
            elev = np.arange(elev_start, elev_levels*elev_step+elev_start, elev_step)
            elev_expand, azim_expand = list(zip(*itertools.product(elev, azim)))
        else:
            elev = np.random.rand(360 // azim_step)*(elev_end-elev_start)+elev_start
            elev_expand, azim_expand = elev.tolist(), azim.tolist()
    else:
        elev_expand, azim_expand = elevs, azims
    assert len(elev_expand) == len(azim_expand)
    R, T = look_at_view_transform(dist, elev_expand, azim_expand, at=at) # positions of cameras are based on the object position (look_at)
    cameras = PerspectiveCameras(focal_length=focal_length, 
                                 principal_point=((w//2, h//2),), 
                                 R=R, 
                                 T=T, 
                                 in_ndc=False, 
                                 image_size=((h, w),),
                                 device=device)
    
    return cameras, azim_expand, elev_expand


def create_ligths(location=((0.0, 3., 0.0),), device="cpu"):
    lights = PointLights(location=location,
                         device=device)
    
    return lights


def create_rasterizer(cameras, img_size, use_naive=False, use_cull_backfaces=False):
    raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0., 
        # faces_per_pixel=20,
        cull_backfaces=use_cull_backfaces, # important for shapenet objects
        bin_size=0 if use_naive else None,
        max_faces_per_bin=50000
    )
    rasterizer = MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
    
    return rasterizer


def create_shader(shader_type, cameras, lights, obj_source="shapenet", device="cpu"):
    if shader_type == "rgb":
        if obj_source in ["shapenet", "pix3d"]:
            Shader = HardFlatShader
        else:
            Shader = HardPhongShader
    elif shader_type == "silhouette":
        Shader = SoftSilhouetteShader
    elif shader_type == "depth":
        Shader = HardDepthShader
    elif shader_type == "normal":
        Shader = HardNormalShader
    elif shader_type == "instances":
        Shader = HardInstanceShader
    
    if shader_type == "silhouette":
        shader = Shader()
    else:
        shader = Shader(
            cameras=cameras,
            lights=lights,
            device=device 
        )
    
    return shader


def create_renderer(rasterizer, shader):
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    
    return renderer


def normal_shading(meshes, fragments) -> torch.Tensor:
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    face_normals = meshes.faces_normals_packed()  # (F, 3)
#     faces_verts = verts[faces]
#     face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel normals
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0

    return pixel_normals


class HardNormalShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = normal_shading(
            meshes=meshes,
            fragments=fragments
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        images = F.normalize(images[..., :3], dim=-1) # To FIX: may need to exclude the last dim during normalization
        # images = F.normalize(images, dim=3)
        return images


def instance_shading(meshes, fragments, obj_meshes) -> torch.Tensor:
    faces_to_mesh_idx = obj_meshes.faces_packed_to_mesh_idx() + 1 # (F,)
    faces_to_mesh_idx = faces_to_mesh_idx.view(-1, 1).expand(faces_to_mesh_idx.shape[0], 3)
#     faces_verts = verts[faces]
#     face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel mesh idx
    faces_to_mesh_idx = faces_to_mesh_idx.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)
    pixel_mesh_idx = faces_to_mesh_idx.gather(0, idx).view(N, H, W, K, 3)
    pixel_mesh_idx[mask] = 0.0
    
    num_meshes = len(obj_meshes.verts_list())
    colors = pixel_mesh_idx / 4

    return colors


class HardInstanceShader(ShaderBase):
    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        obj_meshes = kwargs.get("obj_meshes", None)
        colors = instance_shading(
            meshes=meshes,
            fragments=fragments,
            obj_meshes=obj_meshes
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        # images = F.normalize(images, dim=3)
        return images


def write_scene_images(out_dir, scene_images, obj_images, num_views):
    for i in range(0, num_views):
        Image.fromarray((scene_images["rgb"][i, ..., :3].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, "rgb", f"rgb_{i}.png"))
        Image.fromarray((scene_images["instances"][i, ..., :3].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, "instances", f"instances_{i}.png"))
        
        depth_image = scene_images["depth"][i, ..., 0]
        bg = depth_image==100
        min_d = depth_image[~bg].min().item()
        max_d = depth_image[~bg].max().item()
        depth_vis = (depth_image - min_d) / (max_d - min_d) * (250 - 50) + 50
        depth_vis[bg] = 0
        depth_image[bg] = 0
        Image.fromarray(depth_vis.cpu().numpy().astype(np.uint8)).save(os.path.join(out_dir, "depth", f"depth_{i}.vis.png"))
        imageio.imwrite(os.path.join(out_dir, "depth", f"depth_{i}.png"), depth_image.cpu().numpy().astype(np.uint16)) 
        
        normal_image = scene_images["normal"][i, ..., :3] / 2 * 255 + 127
        Image.fromarray(normal_image.cpu().numpy().astype(np.uint8)).save(os.path.join(out_dir, "normal", f"normal_{i}.png"))
        
        for idx, _ in enumerate(obj_images):
            Image.fromarray((obj_images[idx][i, ..., :3].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, "objects", f"{idx}_{i}.rgb.png"))
            Image.fromarray(obj_images[idx][i, ..., 3].cpu().numpy().astype(np.uint8) != 0).save(os.path.join(out_dir, "objects", f"{idx}_{i}.mask.png"))