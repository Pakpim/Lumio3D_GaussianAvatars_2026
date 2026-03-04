#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from typing import Union
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene import GaussianModel, FlameGaussianModel
from utils.sh_utils import eval_sh

# LM3D : some more arguments
def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, backface_culling = False, depth_map = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    torch.cuda.empty_cache()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=torch.float32, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.cuda(),
        projmatrix=viewpoint_camera.full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.cuda(),
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color


    # --------------------------------------------------------------------
#     #### back-face culling
#     faces = pc.flame_model.faces # (1, 10144, 3)
#     # print("debug faces",faces.shape, faces)
#     verts = pc.verts # (1, 5023, 3)
#     # print("debug verts",verts.shape, verts)
#     triangles = verts[:, faces] # (1, 10144, 3, 3)
#     # print("debug triangles",triangles.shape, triangles)
#     v0 = triangles[:, :, 0, :]  # (1, 10144, 3)
#     v1 = triangles[:, :, 1, :]  # (1, 10144, 3)
#     v2 = triangles[:, :, 2, :]  # (1, 10144, 3)
#     edge1 = v1 - v0  # (1, 10144, 3)
#     edge2 = v2 - v0  # (1, 10144, 3)
#     face_norm = torch.cross(edge1, edge2, dim=-1)  # (1, 10144, 3)
#     face_norm = torch.nn.functional.normalize(face_norm, p=2, dim=-1)
#     # print("debug normal:",type(face_norm), face_norm.shape, face_norm[0][0])
#     camera_center = viewpoint_camera.camera_center.clone().to(pc.get_xyz.device)
#     # print ("debug xyz", pc.get_xyz.device, "\ndebug camera",camera_center.device, camera_center , camera_center.repeat(pc.get_features.shape[0], 1).shape)
#     dir_pp = (pc.get_xyz - camera_center.repeat(pc.get_features.shape[0], 1))
#     dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (1, pc, 3)
#     pc_bind = pc.binding # (pc)
#     pc_norm = face_norm[:, pc_bind] # (1, pc, 3)
#     # print ("debug bind", pc_bind.shape, "debug pc_norm", pc_norm.shape)
#     # print ("debug norm", face_norm.shape, "\ndebug dir", dir_pp_normalized.shape, dir_pp_normalized)
#     dot = (pc_norm[0] * dir_pp_normalized).sum(dim = 1) # (pc)
#     # print ("debug dot", dot.shape)
#     # front = dot <= 0.4 # (pc)
#     front = torch.sigmoid(dot * -10.)
#     # front = torch
#     # print ("debug front", front.shape, front)
#     opacity_back_cull = opacity.clone()
#     # opacity_back_cull[:, 0] *= front # (pc, 1)
#     opacity_ = opacity_back_cull * front.unsqueeze(1) 
    # --------------------------------------------------------------------

    opacity_ = opacity.clone()

    # --------------------------------------------------------------------
    # LM3D : Bcull
    if backface_culling and isinstance(pc, FlameGaussianModel):
        xyz_ , triangles = pc.get_xyz, pc.triangles

        # normal = cross(edge1, edge2)
        normals = torch.cross(triangles[:, :, 1, :] - triangles[:, :, 0, :], 
                            triangles[:, :, 2, :] - triangles[:, :, 0, :], dim=-1)
        normals = normals / normals.norm(dim=1, keepdim=True)

        point_cam = viewpoint_camera.camera_center.repeat(xyz_.shape[0], 1).cuda()

        face_to_cam = point_cam - xyz_
        face_to_cam = face_to_cam / face_to_cam.norm(dim=1, keepdim=True)

        # indexing normals to guass
        normals = normals[:, pc.binding].squeeze(0)

        dot_product = (normals * face_to_cam).sum(dim=1, keepdim=True)

        visible = dot_product >= 0.0

        opacity_ = opacity_ * visible

    # LM3D : depth map
    if depth_map:
        distance = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda()).norm(dim=1, keepdim=True)
        # normalize [0, 1] for opacity
        opacity_ = opacity_ * (1.0 - torch.clamp_min(distance / 10.0, 0.0))
    # --------------------------------------------------------------------
    

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity_,
        scales = scales,
        rotations = rotations, 
        cov3D_precomp = cov3D_precomp)

    visibility = radii > 0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : visibility,
            "radii": radii}
