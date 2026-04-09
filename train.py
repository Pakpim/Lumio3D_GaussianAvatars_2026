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

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' ## for debugging a cuda error but so slow;-;

import torch
from PIL import Image, ImageFile
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_fastgs, network_gui, FASTGS_AVAILABLE

# CUDA fused SSIM (optional, used with FastGS path)
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False
from mesh_renderer import NVDiffRenderer
import sys
from scene import Scene, GaussianModel, FlameGaussianModel

from utils.general_utils import safe_state,save_config

import uuid
from tqdm import tqdm
from utils.image_utils import psnr, error_map
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from PIL import Image
import numpy as np
from utils.general_utils import safe_state, save_config

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def save_image_debug(image, dir, iteration):
    if not os.path.exists(dir):
        os.makedirs(dir)
    image = (image * 255).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(os.path.join(dir, f"iter_{iteration}.jpg"))

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    save_config(dataset, opt, pipe)
    if dataset.bind_to_mesh:
        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.coord, opt, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params, dataset.texture_path)
        mesh_renderer = NVDiffRenderer()
    else:
        gaussians = GaussianModel(dataset.sh_degree, dataset.coord)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.scale_res], opt=opt)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    ####
    bg_color = [0, 0, 0] if dataset.white_background else [0, 0, 0]
    # bg_color = [0.1725,0.3725,0.4588] if dataset.white_background else [0, 0, 0]
    # bg_color = [0.2824,0.4549,0.5294] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    use_fastgs = bool(opt.use_fastgs and FASTGS_AVAILABLE)
    use_fused_ssim = bool(use_fastgs and FUSED_SSIM_AVAILABLE)
    fastgs_filter_interval = max(1, int(getattr(opt, "fastgs_filter_interval", 1)))

    if opt.use_fastgs and not FASTGS_AVAILABLE:
        print("[WARN] --use_fastgs requested, but FastGS extension is unavailable. Falling back to vanilla rasterizer.")
    elif use_fastgs:
        print(f"[INFO] FastGS rasterizer enabled (mult={opt.mult}).")

    if use_fused_ssim:
        print("[INFO] Fused SSIM enabled.")
    elif use_fastgs and not FUSED_SSIM_AVAILABLE:
        print("[WARN] fused_ssim is unavailable. Falling back to standard SSIM.")

    if use_fastgs and fastgs_filter_interval > 1:
        print(f"[INFO] FastGS filter render interval: every {fastgs_filter_interval} iterations.")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    loader_camera_train = DataLoader(scene.getTrainCameras(scale=dataset.scale_res), batch_size=None, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    iter_camera_train = iter(loader_camera_train)
    # viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):   
        torch.cuda.empty_cache()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                # receive data
                net_image = None
                # custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, use_original_mesh = network_gui.receive()
                custom_cam, msg = network_gui.receive()

                # render
                if custom_cam != None:
                    # mesh selection by timestep
                    if gaussians.binding != None:
                        gaussians.select_mesh_by_timestep(custom_cam.timestep, msg['use_original_mesh'])
                    
                    # gaussian splatting rendering
                    if msg['show_splatting']:
                        net_image = render(custom_cam, gaussians, pipe, background, msg['scaling_modifier'])["render"]
                    
                    # mesh rendering
                    if gaussians.binding != None and msg['show_mesh']:
                        out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, gaussians.flame_model.verts_uvs, gaussians.flame_model.textures_idx, gaussians.flame_model.tex_painted, gaussians.flame_model._tex_alpha, custom_cam)
                        print("debug mesh rendering:", out_dict.keys(), out_dict['rgba'].shape)
                        rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                        rgb_mesh = rgba_mesh[:3, :, :]
                        alpha_mesh = rgba_mesh[3:, :, :]

                        # mesh_opacity = msg['mesh_opacity']
                        # if net_image is None:
                        #     net_image = rgb_mesh
                        # else:
                        #     net_image = rgb_mesh * alpha_mesh * mesh_opacity  + net_image * (alpha_mesh * (1 - mesh_opacity) + (1 - alpha_mesh))
                        net_image = rgb_mesh * alpha_mesh

                    # send data
                    ####
                    # out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, gaussians.flame_model.verts_uvs, gaussians.flame_model.textures_idx, gaussians.flame_model.tex_painted, gaussians.flame_model._tex_alpha, custom_cam)
                    # rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
                    # rgba_mesh = rgba_mesh.cuda()
                    net_dict = {'num_timesteps': gaussians.num_timesteps, 'num_points': gaussians._xyz.shape[0]}
                    network_gui.send(net_image, net_dict)
                    # network_gui.send(rgba_mesh[:3,:,:], net_dict)
                if msg['do_training'] and ((iteration < int(opt.iterations)) or not msg['keep_alive']):
                    break
            except Exception as e:
                # print(e)
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # LM3D : one up SH degree every 100 iterations

        if iteration % 100 == 0:
            gaussians.oneupSHdegree()

        try:
            viewpoint_cam = next(iter_camera_train)
        except StopIteration:
            iter_camera_train = iter(loader_camera_train)
            viewpoint_cam = next(iter_camera_train)

        if gaussians.binding != None:
            gaussians.select_mesh_by_timestep(viewpoint_cam.timestep)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        if use_fastgs:
            render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, background, backface_culling=opt.bcull, mult=opt.mult)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, backface_culling=opt.bcull)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        ####
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
      
        # LM3D : alpha map for image
        alpha_map = viewpoint_cam.fg_mask.cuda() # (w, h)
        
        alpha_map = alpha_map.unsqueeze(0).expand_as(gt_image)  # (3, w, h)
        gt_image = gt_image * alpha_map # (3, w, h)

        gt_filter = alpha_map

        compute_filter_render = (not use_fastgs) or (iteration % fastgs_filter_interval == 0)
        filter_render = None
        if compute_filter_render:
            override_color = torch.ones_like(gaussians._xyz).cuda()
            if use_fastgs:
                filter_render = render_fastgs(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    torch.zeros(3).cuda(),
                    override_color=override_color,
                    backface_culling=opt.bcull,
                    depth_map=opt.depth,
                    mult=opt.mult,
                )["render"]
            else:
                filter_render = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    torch.zeros(3).cuda(),
                    override_color=override_color,
                    backface_culling=opt.bcull,
                    depth_map=opt.depth,
                )["render"]

        if opt.disable_gaussian_splats:
            image = torch.zeros_like(image)
            if filter_render is not None:
                filter_render = torch.zeros_like(alpha_map)
       
        if opt.with_texture and iteration >= opt.texture_start_iter:
            out_dict = mesh_renderer.render_from_camera(gaussians.verts, gaussians.faces, gaussians.flame_model.verts_uvs, gaussians.flame_model.textures_idx, gaussians.flame_model._tex_painted, gaussians.flame_model._tex_alpha, viewpoint_cam)
            rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, W, H)
            rgb_mesh = rgba_mesh[:3, :, :]
            alpha_mesh = rgba_mesh[3:, :, :]
            # print("debug texture", image.shape, rgba_mesh.shape,torch.max(image_s[3:, :, :]), torch.min(image_s[3:, :, :]),torch.max(alpha_mesh), torch.min(alpha_mesh))
            image += rgb_mesh * (1 - alpha_map)
            if filter_render is not None:
                filter_render += alpha_mesh * (1 - alpha_map)


        if iteration % pipe.interval_media == 0 or iteration < 3:
            save_image_debug(gt_filter, os.path.join(dataset.model_path, "gt_filter"), iteration)
            if filter_render is not None:
                save_image_debug(filter_render, os.path.join(dataset.model_path, "ren_filter"), iteration)
            save_image_debug(gt_image, os.path.join(dataset.model_path, "gt_image"), iteration)
            save_image_debug(image, os.path.join(dataset.model_path, "rendered"), iteration)

        losses = {}
        losses['l1'] = l1_loss(image, gt_image) * (1.0 - opt.lambda_dssim)
        # Use CUDA fused SSIM when available with FastGS path
        if use_fused_ssim:
            losses['ssim'] = (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))) * opt.lambda_dssim
        else:
            losses['ssim'] = (1.0 - ssim(image, gt_image)) * opt.lambda_dssim


        if opt.train_texture and iteration >= opt.texture_start_iter:
            gt_texture = gaussians.flame_model.input_texture.cuda()
            texture = gaussians.flame_model._tex_painted.cuda()
            texture_alpha = gaussians.flame_model._tex_alpha.cuda()
            losses['texture'] = l1_loss(texture, gt_texture) * opt.texture_lambda

        # LM3D : filter loss
        if filter_render is not None:
            filter_lambda = opt.lambda_filter * (fastgs_filter_interval if use_fastgs else 1)
            losses['filter'] = F.l1_loss(filter_render, gt_filter) * filter_lambda


        if gaussians.binding != None:
            if opt.metric_xyz:
                losses['xyz'] = F.relu((gaussians._xyz*gaussians.face_scaling[gaussians.binding])[visibility_filter] - opt.threshold_xyz).norm(dim=1).mean() * opt.lambda_xyz
            else:
                # losses['xyz'] = gaussians._xyz.norm(dim=1).mean() * opt.lambda_xyz
                losses['xyz'] = F.relu(gaussians._xyz[visibility_filter].norm(dim=1) - opt.threshold_xyz).mean() * opt.lambda_xyz

            if opt.lambda_scale != 0:
                if opt.metric_scale:
                    losses['scale'] = F.relu(gaussians.get_scaling[visibility_filter] - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale
                else:
                    # losses['scale'] = F.relu(gaussians._scaling).norm(dim=1).mean() * opt.lambda_scale
                    losses['scale'] = F.relu(torch.exp(gaussians._scaling[visibility_filter]) - opt.threshold_scale).norm(dim=1).mean() * opt.lambda_scale

            if opt.lambda_dynamic_offset != 0:
                losses['dy_off'] = gaussians.compute_dynamic_offset_loss() * opt.lambda_dynamic_offset

            if opt.lambda_dynamic_offset_std != 0:
                ti = viewpoint_cam.timestep
                t_indices =[ti]
                if ti > 0:
                    t_indices.append(ti-1)
                if ti < gaussians.num_timesteps - 1:
                    t_indices.append(ti+1)
                losses['dynamic_offset_std'] = gaussians.flame_param['dynamic_offset'].std(dim=0).mean() * opt.lambda_dynamic_offset_std
        
            if opt.lambda_laplacian != 0:
                losses['lap'] = gaussians.compute_laplacian_loss() * opt.lambda_laplacian
        
        losses['total'] = sum([v for k, v in losses.items()])
        losses['total'].backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * losses['total'].item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                postfix = {"n_Points": f"{gaussians._xyz.shape[0]:,}"}
                postfix["Loss"] = f"{ema_loss_for_log:.{7}f}"
                # ------------------------
                postfix["opac"] = f"{gaussians.get_opacity.mean().item():.{7}f}"
                if 'filter' in losses:
                    postfix["filter"] = f"{losses['filter']:.{7}f}"
                # ------------------------
                if 'xyz' in losses:
                    postfix["xyz"] = f"{losses['xyz']:.{7}f}"
                if 'scale' in losses:
                    postfix["scale"] = f"{losses['scale']:.{7}f}"
                if 'dy_off' in losses:
                    postfix["dy_off"] = f"{losses['dy_off']:.{7}f}"
                if 'lap' in losses:
                    postfix["lap"] = f"{losses['lap']:.{7}f}"
                if 'dynamic_offset_std' in losses:
                    postfix["dynamic_offset_std"] = f"{losses['dynamic_offset_std']:.{7}f}"

                progress_bar.set_postfix(postfix)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, losses, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (dataset ,pipe, background))
            if (iteration in saving_iterations):
                print("[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration,opt=opt)
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # LM3D : adjust opacity threshold
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.001, scene.cameras_extent, size_threshold)


                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    ###

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            gaussians.clamp_scaling(max_scaling=opt.max_scaling)  # LM3D : clamp scaling to 0.5

            if (iteration in checkpoint_iterations):
                print("[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, losses, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    dataset = renderArgs[0]
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', losses['l1'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', losses['ssim'].item(), iteration)
        if 'xyz' in losses:
            tb_writer.add_scalar('train_loss_patches/xyz_loss', losses['xyz'].item(), iteration)
        if 'scale' in losses:
            tb_writer.add_scalar('train_loss_patches/scale_loss', losses['scale'].item(), iteration)
        if 'dynamic_offset' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset', losses['dynamic_offset'].item(), iteration)
        if 'laplacian' in losses:
            tb_writer.add_scalar('train_loss_patches/laplacian', losses['laplacian'].item(), iteration)
        if 'dynamic_offset_std' in losses:
            tb_writer.add_scalar('train_loss_patches/dynamic_offset_std', losses['dynamic_offset_std'].item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', losses['total'].item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        print("[ITER {}] Evaluating".format(iteration))
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'val', 'cameras' : scene.getValCameras(scale=dataset.scale_res)},
            {'name': 'test', 'cameras' : scene.getTestCameras(scale=dataset.scale_res)},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                print("debug: config['cameras'] = ",config['cameras'], "\n len = ",len(config['cameras']))
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                num_vis_img = 1 # LM3D : 10 to 1
                image_cache = []
                gt_image_cache = []
                vis_ct = 0
                for idx, viewpoint in tqdm(enumerate(DataLoader(config['cameras'], shuffle=False, batch_size=None, num_workers=8)), total=len(config['cameras'])):
                    if scene.gaussians.num_timesteps > 1:
                        scene.gaussians.select_mesh_by_timestep(viewpoint.timestep)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, renderArgs[1], torch.zeros(3).cuda())["render"], 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    ####
                    # override_color = torch.ones_like(self.gaussians._xyz).cuda()
                    # background_color = torch.tensor(self.cfg.background_color).cuda() * 0
                    # alpha_map = render(cam, self.gaussians, self.cfg.pipeline, background_color, scaling_modifier=dpg.get_value("_slider_scaling_modifier"), override_color=override_color)["render"].permute(1, 2, 0).contiguous()
                    # alpha_map = alpha_map[3:, :, :]

                    # Loss
                    if gt_image.shape[0]>3: gt_alpha_map = gt_image[3:, :, :]
                    else: gt_alpha_map = torch.zeros_like(gt_image[:3, :, :])
                    gt_image = gt_image[:3, :, :] * gt_alpha_map
                    image *= gt_alpha_map

                    # print('debug: num_vis_img =',num_vis_img)
                    # print("debug: config['cameras'] = ",config['cameras'], "\n len = ",len(config['cameras']))
                    if tb_writer and (idx % (len(config['cameras']) // num_vis_img) == 0):
                        tb_writer.add_images(config['name'] + "_{}/render".format(vis_ct), image[None], global_step=iteration)
                        error_image = error_map(image, gt_image)
                        tb_writer.add_images(config['name'] + "_{}/error".format(vis_ct), error_image[None], global_step=iteration)
                        ####
                        # tb_writer.add_images(config['name'] + "_{}/alpha_map".format(vis_ct), alpha_map[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(vis_ct), gt_image[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_{}/gt_alphamap".format(vis_ct), gt_alpha_map[None], global_step=iteration)
                        vis_ct += 1
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    ####
                    # image_cache.append(image)
                    # gt_image_cache.append(gt_image)

                    # if idx == len(config['cameras']) - 1 or len(image_cache) == 16:
                    #     batch_img = torch.stack(image_cache, dim=0)
                    #     batch_gt_img = torch.stack(gt_image_cache, dim=0)
                    #     lpips_test += lpips(batch_img, batch_gt_img).sum().double()
                    #     image_cache = []
                    #     gt_image_cache = []

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                # lpips_test /= len(config['cameras'])          
                ssim_test /= len(config['cameras'])          
                print("[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.4f} SSIM {:.4f} LPIPS {:.4f}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def save_image_debug(image, dir, iteration):
    if not os.path.exists(dir):
        os.makedirs(dir)
    image = (image * 255).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(os.path.join(dir, f"iter_{iteration}.jpg"))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--interval", type=int, default=60_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--interval", type=int, default=2_000, help="A shared iteration interval for test and saving results and checkpoints.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    if args.interval > op.iterations:
        args.interval = op.iterations // 5
    if len(args.test_iterations) == 0:
        # args.test_iterations.extend(list(range(2)))
        args.test_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.save_iterations) == 0:
        # args.save_iterations.extend(list(range(2)))
        args.save_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    if len(args.checkpoint_iterations) == 0:
        args.checkpoint_iterations.extend(list(range(args.interval, args.iterations+1, args.interval)))
    
    # args.save_iterations.extend(list(range(2)))
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    ####
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
