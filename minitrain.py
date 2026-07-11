import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import json
import math
import numpy as np
import torch
from PIL import Image
from argparse import ArgumentParser, Namespace
import gc
# These imports assume you have placed this script in the main repository directory
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, FlameGaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state

def save_image_debug(image, path):
    if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path))
    image = (image * 255).clamp(0, 255).byte()
    image = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(image).save(path)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top    =  tanHalfFovY * znear
    bottom = -top
    right  =  tanHalfFovX * znear
    left   = -right
    P = torch.zeros(4, 4, dtype=torch.float32)
    P[0, 0] =  2.0 * znear / (right - left)
    P[1, 1] =  2.0 * znear / (top - bottom)
    P[0, 2] =  (right + left) / (right - left)
    P[1, 2] =  (top + bottom) / (top - bottom)
    P[3, 2] =  1.0
    P[2, 2] =  zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class DummyCamera:
    """A synthetic frontal camera for quick sanity checks."""
    def __init__(self):
        self.image_width = 512
        self.image_height = 512
        self.FoVx = 0.8
        self.FoVy = 0.8
        self.znear = 0.01
        self.zfar  = 100.0

        # Basic frontal camera view (Z=2.0 looking at origin)
        self.world_view_transform = torch.tensor([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  2, 1]
        ], dtype=torch.float32, device="cuda")

        self.projection_matrix = getProjectionMatrix(self.znear, self.zfar, self.FoVx, self.FoVy).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class CameraFromJson:
    """
    Reconstructs a camera from one entry in cameras.json.

    cameras.json format:
      position  – camera position in world space  (3,)
      rotation  – 3x3 matrix: each row is a column of the W2C rotation,
                  i.e. the matrix is stored as R^T (row-major C2W-rotation
                  transposed), so R_w2c = rotation^T
      fx, fy    – focal lengths in pixels
      width, height – image resolution in pixels
    """
    def __init__(self, cam_entry: dict, znear: float = 0.01, zfar: float = 100.0):
        self.image_width  = int(cam_entry["width"])
        self.image_height = int(cam_entry["height"])
        self.znear = znear
        self.zfar  = zfar

        fx = float(cam_entry["fx"])
        fy = float(cam_entry["fy"])
        self.FoVx = 2.0 * math.atan(self.image_width  / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(self.image_height / (2.0 * fy))

        # cameras.json stores the rotation as a 3x3 matrix where the
        # *rows* are the columns of the C2W rotation (i.e. it is R_c2w^T).
        # So R_w2c = np.array(rotation)   (already in W2C row-major form).
        R_w2c = np.array(cam_entry["rotation"], dtype=np.float64)  # (3,3)
        t_world = np.array(cam_entry["position"],  dtype=np.float64)  # (3,) – camera position in world

        # Translation in camera space: t_cam = -R_w2c @ t_world
        t_cam = -R_w2c @ t_world  # (3,)

        # Build the 4x4 W2C (view) matrix in column-major (OpenGL / CUDA rasteriser convention).
        # world_view_transform is stored **transposed** compared to the usual row-major W2C:
        #   world_view_transform[i, j] = W2C[j, i]
        Rt = np.zeros((4, 4), dtype=np.float64)
        Rt[:3, :3] = R_w2c.T          # transpose of W2C rotation → columns = R_w2c rows
        Rt[:3,  3] = t_cam            # translation column
        Rt[ 3,  3] = 1.0
        # world_view_transform = Rt transposed (column-major layout expected by the renderer)
        wvt = torch.tensor(Rt.T, dtype=torch.float32).cuda()

        self.world_view_transform = wvt
        self.projection_matrix    = getProjectionMatrix(znear, zfar, self.FoVx, self.FoVy).cuda()
        self.full_proj_transform  = (wvt.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center        = wvt.inverse()[3, :3]

def spoof_render(dataset, pipe, opt, checkpoint, src_path, output_path, render_mesh, use_dummy_cam,
                 cameras_json=None, cam_idx=0,
                 cam_indices=None, frame_indices=None):
    # 1. Initialize Flame Gaussian Model
    gaussians = FlameGaussianModel(
        dataset.sh_degree, dataset.coord, opt, 
        dataset.disable_flame_static_offset, 
        dataset.not_finetune_flame_params, 
        dataset.texture_path
    )
    
    # 2. Setup Scene or Dummy Camera
    dataset.source_path = src_path
    if not use_dummy_cam and dataset.source_path and dataset.source_path != "./" and dataset.source_path != ".":
        # If we skip checkpoint/ply, this will initialize from scratch
        scene = Scene(
            dataset, 
            gaussians, 
            load_iteration=-1 if checkpoint else 0, 
            shuffle=False, 
            resolution_scales=[dataset.scale_res], 
            ply_path=dataset.ply_path, 
            opt=opt
        )
        print("cam amt" + str(len(scene.getTrainCameras(scale=1))))
    elif cameras_json is not None:
        print(f"Loading camera {cam_idx} from {cameras_json}...")
        with open(cameras_json, 'r') as f:
            cam_list = json.load(f)
        if cam_idx >= len(cam_list):
            raise ValueError(f"cam_idx {cam_idx} out of range (file has {len(cam_list)} cameras)")
        cam = CameraFromJson(cam_list[cam_idx])
        print(f"  img_name={cam_list[cam_idx]['img_name']}  "
              f"res={cam.image_width}x{cam.image_height}  "
              f"FoVx={math.degrees(cam.FoVx):.1f}°  FoVy={math.degrees(cam.FoVy):.1f}°")
    else:
        print("Using a Dummy Camera (Skipping Scene Initialization)!")
        cam = DummyCamera()
    
    # 3. Load Checkpoint
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Restored from checkpoint {checkpoint}")
    else:
        print("Skipping checkpoint load. Gaussians will be in their initial untrained state unless a ply was loaded by Scene.")

    # ------------------------------------------------------------------ #
    # 4. Build the list of (camera, timestep) pairs to render
    # ------------------------------------------------------------------ #
    # Scene mode (source_path set):
    #   - Groups all cameras by timestep.
    #   - cam_indices  : which camera slots to render (default = all)
    #   - frame_indices: which timesteps to render    (default = all)
    #   - cam_stride   : fallback thinning when no lists are given
    # cameras.json mode:
    #   - Uses the one camera built earlier.
    #   - frame_indices filters which FLAME timesteps to render.
    # DummyCamera: single render.
    # ------------------------------------------------------------------ #

    # if 'scene' in dir():  # Scene was loaded in section 2
    #     all_cams = scene.getTrainCameras(scale=1)

    #     # Unique timesteps
    #     sorted_timesteps = sorted({int(c.timestep) for c in all_cams})

    #     # Cameras per timestep
    #     first_ts = sorted_timesteps[0]
    #     cams_per_ts = sum(1 for c in all_cams if int(c.timestep) == first_ts)

    #     print(f"{len(sorted_timesteps)} timesteps")
    #     print(f"{cams_per_ts} cams per timestep")
    #     if frame_indices is not None:
    #         ts_to_render = [sorted_timesteps[f] for f in frame_indices]
    #     else:
    #         ts_to_render = sorted_timesteps

    #     if cam_indices is not None:
    #         slots_to_render = cam_indices
    #     elif cam_stride > 1:
    #         slots_to_render = list(range(0, cams_per_ts, cam_stride))
    #     else:
    #         slots_to_render = list(range(cams_per_ts))

    # ------------------------------------------------------------------ #
    # 5. Render loop
    # ------------------------------------------------------------------ #
    from mesh_renderer import NVDiffRenderer
    mesh_renderer = NVDiffRenderer() if render_mesh else None

    bg_color = [0, 0, 0]  # swap to [1,1,1] for white background
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Background set")

    os.makedirs(output_path, exist_ok=True)
    if frame_indices is not None:
        ts_to_render = frame_indices
    else:
        ts_to_render = range(0, 15)

    all_slots = [1,3,4,7,9,10,11,12,14,15,16,19,20,26,29,31,32,34,36,38,40,42,43,44,45,46,48,51,52,58,61,63]
    if cam_indices is not None:
        slots_to_render = range(len(all_slots))
    else:
        slots_to_render = range(len(all_slots))

    print(ts_to_render, slots_to_render)

    total_jobs = len(ts_to_render) * len(slots_to_render)
    counter = 0

    all_cams = scene.getTrainCameras(scale=1)
    cams_per_ts = len(all_slots)
    print(cams_per_ts)

    for ts in ts_to_render:
        for slot in slots_to_render:
            print(ts, slot)
            cam = all_cams[ts * cams_per_ts + slot]
            timestep = ts
            # Update the FLAME mesh for this timestep (skip when no timestep info)
            if timestep is not None and gaussians.binding is not None:
                gaussians.select_mesh_by_timestep(timestep)
                ts_tag = f"t{timestep:04d}"
            else:
                ts_tag = "t0000"

            cam_tag = getattr(cam, 'image_name', str(counter))
            outname = os.path.join(output_path, f"spoof_{ts_tag}_{cam_tag[-5:]}.jpg")

            print(ts_tag, cam_tag)
            print(f"[{counter+1}/{len(ts_to_render) * len(slots_to_render)}] timestep={timestep}  cam={cam_tag[-5:]}  -> {outname}")
            counter += 1
            with torch.no_grad():
                if render_mesh:
                    print("this")
                    out_dict = mesh_renderer.render_from_camera(
                        gaussians.verts,
                        gaussians.faces,
                        gaussians.flame_model.verts_uvs,
                        gaussians.flame_model.textures_idx,
                        gaussians.flame_model._tex_painted,
                        gaussians.flame_model._tex_alpha,
                        cam,
                    )
                    rgba_mesh = out_dict['rgba'].squeeze(0).permute(2, 0, 1)  # (C, H, W)
                    rgb_mesh  = rgba_mesh[:3]
                    alpha_mesh = rgba_mesh[3:]
                    image = rgb_mesh * alpha_mesh

                    del rgba_mesh
                    del rgb_mesh
                    del alpha_mesh
                    del out_dict
                else:
                    print("that")
                    render_pkg = render(cam, gaussians, pipe, background)
                    image = render_pkg["render"]

            save_image_debug(image, outname)

            # del image
            # del render_pkg
            print(f"  Saved -> {outname}")
            # gc.collect()
            # torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Spoof Render Script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--checkpoint",     type=str,   default=None,           help="Path to the trained model checkpoint (.pth)")
    parser.add_argument("--src_path",       type=str,   default=None,           help="Path to the source directory")
    parser.add_argument("--output_path",    type=str,   default="spoof_render", help="Output directory for rendered images")
    parser.add_argument("--render_mesh",    action="store_true",                help="Render the raw FLAME mesh instead of splats")
    parser.add_argument("--dummy_cam",      action="store_true",                help="Force the use of a dummy camera and skip dataset/Scene loading entirely")
    parser.add_argument("--cameras_json",   type=str,   default=None,           help="Path to cameras.json; uses this camera instead of dummy/scene")
    parser.add_argument("--cam_idx",        type=int,   default=0,              help="(cameras.json mode) index of the single camera to load")
    parser.add_argument("--cam_indices",    type=int,   nargs='+', default=None,
                        help="(Scene mode) camera slot indices to render, e.g. --cam_indices 0 7 15. Renders ALL frames for each. THIS DOESN'T WORK. I JUST MAKE IT DO NOTHING.")
    parser.add_argument("--frame_indices",  type=int,   nargs='+', default=None,
                        help="Timestep indices to render, e.g. --frame_indices 0 10 20. Works in Scene and cameras.json modes.")
    parser.add_argument("--quiet",          action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    
    safe_state(args.quiet)
    
    spoof_render(
        dataset=lp.extract(args),
        pipe=pp.extract(args),
        opt=op.extract(args),
        checkpoint=args.checkpoint,
        src_path=args.src_path,
        output_path=args.output_path,
        render_mesh=args.render_mesh,
        use_dummy_cam=args.dummy_cam,
        cameras_json=args.cameras_json,
        cam_idx=args.cam_idx,
        cam_indices=args.cam_indices,
        frame_indices=args.frame_indices,
    )
