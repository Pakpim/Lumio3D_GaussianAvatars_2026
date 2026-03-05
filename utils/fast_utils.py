"""
FastGS utilities adapted for Lumio3D GaussianAvatars.

Provides multi-view camera sampling and Gaussian scoring functions
used by the FastGS multiview-consistent densification pipeline.
"""

import torch
import random
from gaussian_renderer import render, FASTGS_AVAILABLE
from .loss_utils import l1_loss, ssim

# Fused SSIM (much faster than standard SSIM)
try:
    from fused_ssim import fused_ssim as fast_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    print("[WARNING] fused-ssim not available, falling back to standard SSIM")
    fast_ssim = None
    FUSED_SSIM_AVAILABLE = False


def sampling_cameras(my_viewpoint_stack, num_cams=10):
    """Randomly sample a given number of cameras from the viewpoint stack."""
    num_cams = min(num_cams, len(my_viewpoint_stack))
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))
    return camlist


def get_loss(reconstructed_image, original_image):
    """Per-pixel L1 loss, normalized to [0, 1]."""
    l1 = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_norm = (l1 - torch.min(l1)) / (torch.max(l1) - torch.min(l1) + 1e-7)
    return l1_norm


def compute_photometric_loss(viewpoint_cam, image):
    """Combined L1 + SSIM photometric loss for a rendered view."""
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_val = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_val = ssim(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim_val)
    return loss


def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args,
                                   use_fastgs=True, mult=0.5, DENSIFY=False):
    """Compute multi-view consistency scores for Gaussians.

    When the FastGS rasterizer is available, uses hardware-accelerated metric
    maps for precise per-Gaussian error counting. Otherwise falls back to a
    gradient-based approximation.

    Args:
        camlist: list of viewpoint cameras to render from.
        gaussians: current Gaussian model (supports FLAME binding).
        pipe: pipeline parameters.
        bg: background tensor.
        args: config with ``loss_thresh`` attribute.
        use_fastgs: whether to use the FastGS rasterizer path.
        mult: FastGS tile multiplier for compact bounding boxes.
        DENSIFY: if True, also compute an importance score for densification.

    Returns:
        importance_score: per-Gaussian counts of flagged views (if DENSIFY).
        pruning_score: normalized [0,1] per-Gaussian reconstruction consistency.
    """
    if use_fastgs and FASTGS_AVAILABLE:
        return _compute_score_fastgs_rasterizer(camlist, gaussians, pipe, bg, args, mult, DENSIFY)
    else:
        return _compute_score_fallback(camlist, gaussians, pipe, bg, args, DENSIFY)


def _compute_score_fastgs_rasterizer(camlist, gaussians, pipe, bg, args, mult, DENSIFY):
    """Score Gaussians using FastGS hardware metric maps."""
    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        # Select FLAME mesh timestep if bound
        if gaussians.binding is not None:
            gaussians.select_mesh_by_timestep(my_viewpoint_cam.timestep)

        render_image = render(my_viewpoint_cam, gaussians, pipe, bg,
                              use_fastgs=True, mult=mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        l1_loss_norm = get_loss(render_image, gt_image)

        loss_thresh = getattr(args, 'loss_thresh', 0.5)
        metric_map = (l1_loss_norm > loss_thresh).int()

        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg,
                            use_fastgs=True, mult=mult,
                            get_flag=True, metric_map=metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (
        torch.max(full_metric_score) - torch.min(full_metric_score) + 1e-7)

    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score


def _compute_score_fallback(camlist, gaussians, pipe, bg, args, DENSIFY):
    """Fallback: score Gaussians using vanilla renderer + gradient accumulation."""
    full_gradient_accum = torch.zeros(gaussians.get_xyz.shape[0], device='cuda')
    full_error_score = torch.zeros(gaussians.get_xyz.shape[0], device='cuda')

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]

        if gaussians.binding is not None:
            gaussians.select_mesh_by_timestep(my_viewpoint_cam.timestep)

        render_pkg = render(my_viewpoint_cam, gaussians, pipe, bg)
        render_image = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]

        gt_image = my_viewpoint_cam.original_image.cuda()
        l1_loss_map = torch.abs(render_image - gt_image).mean(dim=0)
        l1_loss_norm = get_loss(render_image, gt_image)

        loss_thresh = getattr(args, 'loss_thresh', 0.5)
        high_error_mask = (l1_loss_norm > loss_thresh).float()
        error_metric = l1_loss_map * high_error_mask

        visible_indices = visibility_filter.squeeze()
        full_error_score[visible_indices] += error_metric.mean()

        if DENSIFY:
            view_gradients = torch.norm(gaussians.xyz_gradient_accum, dim=-1)
            full_gradient_accum += view_gradients

    pruning_score = (full_error_score - full_error_score.min()) / (
        full_error_score.max() - full_error_score.min() + 1e-7)

    if DENSIFY:
        importance_score = (full_gradient_accum / len(camlist)).long()
    else:
        importance_score = None

    return importance_score, pruning_score
