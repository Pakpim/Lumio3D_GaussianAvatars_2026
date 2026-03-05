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

try:
    from diff_gaussian_rasterization_fastgs import (
        GaussianRasterizationSettings as FastGSRasterSettings,
        GaussianRasterizer as FastGSRasterizer,
    )
    FASTGS_AVAILABLE = True
except ImportError:
    FASTGS_AVAILABLE = False


# LM3D : some more arguments
def render(viewpoint_camera, pc : Union[GaussianModel, FlameGaussianModel], pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, backface_culling = False, depth_map = False,
           use_fastgs=False, mult=0.5, get_flag=False, metric_map=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if use_fastgs and FASTGS_AVAILABLE:
        return _render_fastgs(viewpoint_camera, pc, pipe, bg_color,
                              scaling_modifier, override_color, backface_culling, depth_map,
                              mult, get_flag, metric_map)
    else:
        return _render_vanilla(viewpoint_camera, pc, pipe, bg_color,
                               scaling_modifier, override_color, backface_culling, depth_map)


def _apply_opacity_modifiers(pc, opacity, viewpoint_camera, means3D, backface_culling, depth_map):
    """Apply backface culling and depth-based opacity modifications (shared by both paths)."""
    opacity_ = opacity.clone()

    # LM3D : Bcull
    if backface_culling and isinstance(pc, FlameGaussianModel):
        xyz_, triangles = pc.get_xyz, pc.triangles

        normals = torch.cross(triangles[:, :, 1, :] - triangles[:, :, 0, :],
                              triangles[:, :, 2, :] - triangles[:, :, 0, :], dim=-1)
        normals = normals / normals.norm(dim=1, keepdim=True)

        point_cam = viewpoint_camera.camera_center.repeat(xyz_.shape[0], 1).cuda()
        face_to_cam = point_cam - xyz_
        face_to_cam = face_to_cam / face_to_cam.norm(dim=1, keepdim=True)

        normals = normals[:, pc.binding].squeeze(0)
        dot_product = (normals * face_to_cam).sum(dim=1, keepdim=True)
        visible = dot_product >= 0.0
        opacity_ = opacity_ * visible

    # LM3D : depth map
    if depth_map:
        distance = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1).cuda()).norm(dim=1, keepdim=True)
        opacity_ = opacity_ * (1.0 - torch.clamp_min(distance / 10.0, 0.0))

    return opacity_


def _compute_sh_and_colors(pc, pipe, viewpoint_camera, override_color):
    """Compute SH coefficients / precomputed colors (shared by both paths)."""
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    return shs, colors_precomp


def _render_vanilla(viewpoint_camera, pc, pipe, bg_color,
                    scaling_modifier=1.0, override_color=None,
                    backface_culling=False, depth_map=False):
    """Original vanilla diff-gaussian-rasterization path."""
    torch.cuda.empty_cache()
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

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs, colors_precomp = _compute_sh_and_colors(pc, pipe, viewpoint_camera, override_color)
    opacity_ = _apply_opacity_modifiers(pc, opacity, viewpoint_camera, means3D, backface_culling, depth_map)

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity_,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    visibility = radii > 0

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": visibility,
            "radii": radii}


def _render_fastgs(viewpoint_camera, pc, pipe, bg_color,
                   scaling_modifier=1.0, override_color=None,
                   backface_culling=False, depth_map=False,
                   mult=0.5, get_flag=False, metric_map=None):
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype,
                                     requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if metric_map is None:
        metric_map = torch.zeros(
            int(viewpoint_camera.image_height) * int(viewpoint_camera.image_width),
            dtype=torch.int, device='cuda')

    raster_settings = FastGSRasterSettings(
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
        mult=mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map=metric_map,
    )

    rasterizer = FastGSRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # FastGS rasterizer uses split dc/rest SH interface
    shs = None
    dc = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            dc = pc.get_features_dc
            shs = pc.get_features_rest
    else:
        colors_precomp = override_color

    opacity_ = _apply_opacity_modifiers(pc, opacity, viewpoint_camera, means3D, backface_culling, depth_map)

    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D=means3D,
        means2D=means2D,
        dc=dc,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity_,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": (radii > 0).nonzero().squeeze(-1),
            "radii": radii,
            "accum_metric_counts": accum_metric_counts}
