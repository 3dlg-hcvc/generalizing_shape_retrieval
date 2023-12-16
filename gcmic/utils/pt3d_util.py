import os, sys
import torch
from PIL import Image
import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    look_at_rotation,
    PerspectiveCameras,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    HardFlatShader,
    TexturesVertex,
    BlendParams
)

from preprocess.moos.obj_util import load_3dfuture_metadata, load_3dfuture_obj, load_pix3d_obj, load_shapenet_metadata, load_shapenet_obj
from preprocess.moos.render_util import create_ligths, create_cameras, create_rasterizer, create_shader, create_renderer
from gcmic.utils.io_util import normalize_verts


def render_obj_in_pix3d_world(obj_names, pose_info, constant_color=[1., 1., 1.], device='cpu'):
    """Render pix3d object in the world coordinate given pose

    Args:
        obj_path (str): _description_
        pose_info (dict): _description_

    Returns:
        tensor: images (N, H, W, 4)
    """
    w, h = pose_info["img_size"]
    focal_length = pose_info["focal_length"]
    sensor_width = pose_info["sensor_width"]
    focal_length *= w / sensor_width
    
    mesh_list = []
    for obj_name in obj_names:
        obj_source = obj_name.split('.')[0]
        obj_id = obj_name.replace(f'{obj_source}.', '')
        verts, faces, _ = load_pix3d_obj(obj_id, load_textures=False, device=device)
        trans3d = torch.tensor(pose_info["trans_mat"]).type_as(verts)
        rot_mat3d = torch.tensor(pose_info["rot_mat"]).type_as(verts)
        verts = (rot_mat3d @ verts.T).T + trans3d
        
        mesh = Meshes(verts=[verts], faces=[faces]).to(device)
        color = torch.ones(1, verts.shape[0], 3, device=device) * torch.tensor(constant_color, device=device).float() # assign constant color texture
        mesh.textures = TexturesVertex(verts_features=color)
        mesh_list.append(mesh)
    batch_meshes = join_meshes_as_batch(mesh_list)
    
    ##############################################################################################
    # Initialize a camera.
    img_size = pose_info["img_size"]
    lights = create_ligths(device=device)
    R = look_at_rotation(((0,0,0),), at=((0,0,1),), up=((0,1,0),)) 
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=((w//2, h//2),), device=device, R=R, in_ndc=False, image_size=((h, w),))

    # Define the settings for rasterization and shading. 
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0., 
        max_faces_per_bin=50000
        # faces_per_pixel=50
    )
    # Use HardFlatShader here since mostly resemble STK netural rendering
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = {"rgb": renderer(batch_meshes)}
    images["mask"] = images["rgb"][..., 3] != 0
    
    return images


def render_obj_in_scan2cad_world(obj_names, pose_info, constant_color=[1., 1., 1.], device='cpu'):
    """Render scan2cad object in the world coordinate given pose

    Args:
        obj_path (str): _description_
        pose_info (dict): _description_

    Returns:
        tensor: images (N, H, W, 4)
    """
    w, h = pose_info["img_size"]
    intrinscis = pose_info["cam_intrinsics"]
    fx, fy, cx, cy = intrinscis[0][0], intrinscis[1][1], intrinscis[0][2], intrinscis[1][2]
    
    mesh_list = []
    for obj_name in obj_names:
        obj_source = obj_name.split('.')[0]
        obj_id = obj_name.replace(f'{obj_source}.', '')
        verts, faces, _ = load_shapenet_obj(obj_id, load_textures=False, device=device)
        trans3d = torch.tensor(pose_info["trans_mat"]).type_as(verts)
        rot_mat3d = torch.tensor(pose_info["rot_mat"]).type_as(verts)
        verts = (rot_mat3d @ verts.T).T + trans3d
        verts[:, [1,2]] *= -1
        
        mesh = Meshes(verts=[verts], faces=[faces]).to(device)
        color = torch.ones(1, verts.shape[0], 3, device=device) * torch.tensor(constant_color, device=device).float() # assign constant color texture
        mesh.textures = TexturesVertex(verts_features=color)
        mesh_list.append(mesh)
    batch_meshes = join_meshes_as_batch(mesh_list)
    
    ##############################################################################################
    # Initialize a camera.
    lights = create_ligths(device=device)
    R = look_at_rotation(((0,0,0),), at=((0,0,-1),), up=((0,1,0),)) 
    cameras = PerspectiveCameras(focal_length=((fx, fy),), principal_point=((cx, cy),), device=device, R=R, in_ndc=False, image_size=((h, w),))

    # Define the settings for rasterization and shading. 
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0., 
        max_faces_per_bin=50000,
        cull_backfaces=True
        # faces_per_pixel=50
    )
    # Use HardFlatShader here since mostly resemble STK netural rendering
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardFlatShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = {"rgb": renderer(batch_meshes)}
    images["mask"] = images["rgb"][..., 3] != 0
    
    return images


_, obj_scales_3dfuture = load_3dfuture_metadata()
_, obj_scales_shapenet = load_shapenet_metadata()
def render_obj_in_moos_world(obj_names, pose_info, shader_types=["rgb"], constant_color=[1., 1., 1.], device='cpu'):
    w, h = pose_info["img_size"]
    focal_length = pose_info["focal_length"]
    sensor_width = pose_info["sensor_width"]
    focal_length *= w / sensor_width

    mesh_list = []
    for obj_name in obj_names:
        obj_source, obj_id = obj_name.split('.')
        if obj_source == "pix3d":
            obj_scales = None
            load_obj = load_pix3d_obj
        elif obj_source == "3dfuture":
            obj_scales = obj_scales_3dfuture
            load_obj = load_3dfuture_obj
        elif obj_source == "shapenet":
            obj_scales = obj_scales_shapenet
            load_obj = load_shapenet_obj
        # try:
        #     verts, faces, tex = load_obj(obj_id, obj_scales=obj_scales, device=device)
        # except:
        verts, faces, _ = load_obj(obj_id, load_textures=False, obj_scales=obj_scales, use_pre_norm=True, device=device)
        if obj_source == "shapenet":
            verts[:, [0,2]] = - verts[:, [0,2]] # HACK to face +z axis
        trans3d = torch.tensor(pose_info["trans_mat"]).type_as(verts)
        rot_mat3d = torch.tensor(pose_info["rot_mat"]).type_as(verts)
        verts = (rot_mat3d @ verts.T).T + trans3d
        verts[:, 1] -= verts[:, 1].min() # make sure object is on the ground
        
        # if tex is None:
        mesh = Meshes(verts=[verts], faces=[faces]).to(device)
        color = torch.ones(1, verts.shape[0], 3, device=device) * torch.tensor(constant_color, device=device).float() # assign constant color texture
        mesh.textures = TexturesVertex(verts_features=color)
        # else:
        #     mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        mesh_list.append(mesh)
    batch_meshes = join_meshes_as_batch(mesh_list)
    
    img_size = pose_info["img_size"]
    lights = create_ligths(device=device)
    cameras, _, _ = create_cameras(dist=pose_info["cam_dist"],
                                    at=(pose_info["cam_look_at"],),
                                    img_size=img_size,
                                    azims=[pose_info["cam_azim"]],
                                    elevs=[pose_info["cam_elev"]],
                                    device=device)
    shaders = {shader_type: create_shader(shader_type, cameras, lights, obj_source=obj_source, device=device) for shader_type in shader_types}
    try:
        rasterizer = create_rasterizer(cameras, img_size=img_size, use_cull_backfaces=obj_source=="shapenet")
        renderers = {shader_type: create_renderer(rasterizer, shaders[shader_type]) for shader_type in shader_types}
        images = {image_type: renderers[image_type](batch_meshes,
                                                    blend_params=BlendParams(background_color=(0., 0., 0.) if image_type=="normal" else (1., 1., 1.))) 
                for image_type in shader_types}
    except:
        rasterizer = create_rasterizer(cameras, img_size=img_size, use_naive=True, use_cull_backfaces=obj_source=="shapenet")
        renderers = {shader_type: create_renderer(rasterizer, shaders[shader_type]) for shader_type in shader_types}
        images = {image_type: renderers[image_type](batch_meshes,
                                                    blend_params=BlendParams(background_color=(0., 0., 0.) if image_type=="normal" else (1., 1., 1.))) 
                for image_type in shader_types}
        
    images["mask"] = images["rgb"][..., 3] != 0
    
    return images


def render_3dfuture_view(obj_id, out_dir=None, azims=None, elevs=None, shader_types=["rgb", "normal"], device='cuda'):
    obj_filename = f"/path_to_3dfront/3D-FUTURE-model/{obj_id}/normalized_model.obj"
    verts, faces, _ = load_obj(obj_filename, load_textures=False)
    verts = normalize_verts(verts, scale_along_diagonal=True)
    mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])
    color = torch.ones(1, verts.size(0), 3, device=device)
    mesh.textures = TexturesVertex(verts_features=color)
    
    img_size = (512, 512)
    lights = create_ligths(device=device)
    cameras, _, _ = create_cameras(dist=1.5,
                                    img_size=img_size,
                                    azims=azims,
                                    elevs=elevs,
                                    device=device)
    num_views = len(azims)
    shaders = {shader_type: create_shader(shader_type, cameras, lights, obj_source="3dfuture", device=device) for shader_type in shader_types}
    rasterizer = create_rasterizer(cameras, img_size=img_size)
    renderers = {shader_type: create_renderer(rasterizer, shaders[shader_type]) for shader_type in shader_types}
    
    batch_meshes = join_meshes_as_batch([mesh] * num_views)
    images = {image_type: renderers[image_type](batch_meshes,
                                                blend_params=BlendParams(background_color=(0., 0., 0.) if image_type=="normal" else (1., 1., 1.))) 
            for image_type in shader_types}
    images["mask"] = images["rgb"][..., 3] != 0
    
    if out_dir:
        os.makedirs(os.path.join(out_dir, obj_id), exist_ok=True)
        for view_idx in range(num_views):
            Image.fromarray((images["rgb"][view_idx, ..., :3].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, obj_id, f"rgb.{view_idx}.png"))
            Image.fromarray((images["mask"][view_idx].cpu().numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, obj_id, f"mask.{view_idx}.png"))
            normal_image = images["normal"][view_idx, ..., :3] / 2 * 255 + 127
            Image.fromarray(normal_image.cpu().numpy().astype(np.uint8)).save(os.path.join(out_dir, obj_id, f"normal.{view_idx}.png"))
    
    return images