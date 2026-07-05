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

from typing import Optional
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
# from pytorch3d.transforms import quaternion_multiply
from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from mesh_renderer import NVDiffRenderer

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, coord):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        ####
        # self.max_sh_degree = 1  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.coord = coord

        # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
        # for binding GaussianModel to a mesh
        self.face_center = None
        self.face_scaling = None
        self.face_orien_mat = None
        self.face_orien_quat = None
        self.binding = None  # gaussian index to face index
        self.binding_counter = None  # number of points bound to each face
        self.timestep = None  # the current timestep
        self.num_timesteps = 1  # required by viewers

        self.normal_offset = None # position offset by pc's normal

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.normal_offset,
            self.binding,
            self.binding_counter,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.normal_offset,
        self.binding,
        self.binding_counter,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args[:15]
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.binding is None:
            return self.scaling_activation(self._scaling)
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_scaling is None:
                self.select_mesh_by_timestep(0)
            
            scaling = self.scaling_activation(self._scaling)
            tmp = scaling * self.face_scaling[self.binding]
            # TODO optimize
            # print("debug scaling", tmp.shape, tmp.dtype)
            # assert False, "exit jaa"
            return scaling * self.face_scaling[self.binding]
    
    @property
    def get_rotation(self):
        if self.binding is None:
            return self.rotation_activation(self._rotation)
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_orien_quat is None:
                self.select_mesh_by_timestep(0)

            # always need to normalize the rotation quaternions before chaining them
            rot = self.rotation_activation(self._rotation)
            face_orien_quat = self.rotation_activation(self.face_orien_quat[self.binding])
            return quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(face_orien_quat), quat_wxyz_to_xyzw(rot)))  # roma
            # return quaternion_multiply(face_orien_quat, rot)  # pytorch3d
    
    @property
    def get_xyz(self):
        if self.binding is None:
            return self._xyz
        else:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            if self.face_center is None:
                self.select_mesh_by_timestep(0)
            
            if self.coord == "bary":
                verts = self.verts # (B, V, 3)
                faces = self.flame_model.faces.int() # (F, 3)
                vert_binding = faces[self.binding] # (PC, 3)
                pc_verts_xyz = verts[:, vert_binding] # (B, PC, 3, 3)
                bary = torch.abs(self._xyz)
                # bary = self._xyz.clamp(min=0.00001)
                assert not torch.isnan(bary).any(), "NaN in barycentric coordinates 2"
                bary = bary/bary.sum(dim=1, keepdim=True) # (PC, 3)
                # print("debug bary:", bary[...].shape, "\n pc_verts_xyz:", pc_verts_xyz.shape)
                b0 = bary[:, 0, None] * pc_verts_xyz[0, :, 0, :] # (PC, 3)
                b1 = bary[:, 1, None] * pc_verts_xyz[0, :, 1, :] # (PC, 3)
                b2 = bary[:, 2, None] * pc_verts_xyz[0, :, 2, :] # (PC, 3)
                # print("debug b0:", b0.shape, "\n debug b1:", b1.shape, "\n debug b2:", b2.shape)

                vnormals = NVDiffRenderer.compute_v_normals(self, verts, faces)  # (B, V, 3)
                pc_verts_normal = vnormals[:, vert_binding]
                n0 = bary[:, 0, None] * pc_verts_normal[0, :, 0, :] # (PC, 3)
                n1 = bary[:, 1, None] * pc_verts_normal[0, :, 1, :] # (PC, 3)
                n2 = bary[:, 2, None] * pc_verts_normal[0, :, 2, :] # (PC, 3)
                return (b0 + b1 + b2) + (n0 + n1 +n2) * self.normal_offset[..., None] # (PC, 3)

            _xyz = torch.tensor(self._xyz.clone(), dtype=torch.float64)
            face_center = torch.tensor(self.face_center[self.binding].clone(), dtype=torch.float64)
            face_scaling = torch.tensor(self.face_scaling[self.binding].clone(), dtype=torch.float64)
            face_orien_mat = torch.tensor(self.face_orien_mat[self.binding].clone(), dtype=torch.float64)
            
            xyz = torch.bmm(face_orien_mat, _xyz[..., None]).squeeze(-1)
            return (xyz * face_scaling + face_center).float()

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        ####
        # print("debug opacity",self._opacity)
        # print("debug opacity activation",self.opacity_activation(self._opacity))
        dbg = self.opacity_activation(self._opacity)
        dbg = torch.ones_like(self._opacity)
        # print("debug opacity:", type(dbg),dbg.shape,dbg)
        return self.opacity_activation(self._opacity)

    # LM3D : clamp scaling
    def clamp_scaling(self, max_scaling=0.5):
        max_scaling = np.log(max_scaling)
        self._scaling = torch.clamp(self._scaling, max=max_scaling)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def select_mesh_by_timestep(self, timestep):
        raise NotImplementedError

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : Optional[BasicPointCloud], spatial_lr_scale : float, initial_pc_size):
        self.spatial_lr_scale = spatial_lr_scale
        if pcd == None:
            assert self.binding is not None
            num_pts = self.binding.shape[0]

            if self.coord == "bary":
                fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda()
                # fused_point_cloud = torch.tensor(np.random.random((num_pts, 3))).float().cuda()
                fused_point_cloud[..., 0:2] = torch.tensor(np.random.random((num_pts, 2))).float().cuda()
                tmp = fused_point_cloud.sum(axis = 1) > 1.0
                fused_point_cloud[tmp,0:2] = 1.0 - fused_point_cloud[tmp,0:2]
                fused_point_cloud[:, 2] = 1.0 - fused_point_cloud[:, 0:2].sum(axis = 1)

            else:
                fused_point_cloud = torch.zeros((num_pts, 3)).float().cuda()


            fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()
        else:
            fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
            
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self.normal_offset = nn.Parameter(torch.zeros_like(self._xyz[..., 0]).requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        print("Number of points at initialisation: ", self.get_xyz.shape[0])

        if self.binding is None:
            dist2 = torch.clamp_min(distCUDA2(self.get_xyz), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        else:

            scales = torch.log(torch.ones((self.get_xyz.shape[0], 3), device="cuda") * initial_pc_size) # LM3D : 0.01 is the initial scale

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # LM3D : opacity set to 1.0 for all points
        opacities = inverse_sigmoid(torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        ####
        l = []
        if (training_args.disable_gaussian_splats):
            l = [
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"}
            ]
        else:
            # print("ok juff")
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self.normal_offset], 'lr': training_args.normal_position_lr, "name": "normal_offset"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            
    # LM3D : reset z (y)
    def reset_z_position(self):
        """ Reset the z position of the points to 0. """
        self._xyz.data[:, 1] = 0.0

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('normal_offset')
        if self.binding is not None:
            for i in range(1):
                l.append('binding_{}'.format(i))
        return l

    def save_ply(self, path, opt):
        mkdir_p(os.path.dirname(path))

        if self.coord == "bary":
            xyz = torch.tensor(self.get_xyz.clone().detach(), dtype=torch.float64)
            xyz_global = xyz.clone()
            face_center = torch.tensor(self.face_center[self.binding].clone().detach(), dtype=torch.float64)
            face_scaling = torch.tensor(self.face_scaling[self.binding].clone().detach(), dtype=torch.float64)
            face_orien_mat = torch.tensor(self.face_orien_mat[self.binding].clone().detach(), dtype=torch.float64)
            eps = 1e-9
            # print("debug xyz", torch.all(torch.isfinite(xyz)))
            assert not torch.isnan(xyz).any(), "NaN in barycentric coordinates 1"
            tmp = xyz.clone() - face_center
            assert torch.all(torch.abs(tmp + face_center - xyz_global) < eps), "Barycentric coordinates do not match original xyz coordinates 1"
            xyz = tmp.clone() / face_scaling
            
            # print("debug face scale",torch.all(face_scaling > 0.0))
            assert torch.all(face_scaling > 0.0), "zero in face scaling"
            assert not (torch.isnan(xyz).any()), "NaN in barycentric coordinates 2"
            # print("debug face scaling", self.face_scaling.shape, self.face_scaling)
            assert torch.all((face_scaling / face_scaling) == 1.0) , "precision?"
            tmp2 = tmp  * (face_scaling / face_scaling)
            assert torch.all(torch.abs(tmp2 - tmp) < eps), "Barycentric coordinates do not match original xyz coordinates 2"
            tmp = xyz * face_scaling + face_center
            assert torch.all(torch.abs(tmp - xyz_global) < eps), "Barycentric coordinates do not match original xyz coordinates 3"
            
            inv = torch.linalg.inv(face_orien_mat)
            xyz = torch.bmm(inv, xyz[..., None]).squeeze(-1)

            # print("debug type", inv.dtype, xyz.dtype, xyz_global.dtype)

            tmp = torch.bmm(face_orien_mat, xyz[..., None]).squeeze(-1)
            tmp = tmp * face_scaling + face_center
            # print("debug max error", torch.max(torch.abs(tmp - xyz_global)))
            assert torch.all(torch.abs(tmp - xyz_global) < eps), "Barycentric coordinates do not match original xyz coordinate 4"
            
            xyz = xyz.detach().cpu().numpy()
        else:
            xyz = self._xyz.detach().cpu().numpy()
        
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        normal_offset = self.normal_offset.detach().cpu().numpy()[:, None]

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, normal_offset), axis=1)

        if self.binding is not None:
            binding = self.binding.detach().cpu().numpy()
            attributes = np.concatenate((attributes, binding[:, None]), axis=1)

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.coord == "bary":
            path = path.replace(".ply", "_bary.ply")
            uvw = self._xyz.detach().cpu().numpy()
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((uvw, normals, f_dc, f_rest, opacities, scale, rotation, normal_offset), axis=1)

            if self.binding is not None:
                binding = self.binding.detach().cpu().numpy()
                attributes = np.concatenate((attributes, binding[:, None]), axis=1)

            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_scaling(self, scale_factor: float):
        scale_factor = float(scale_factor)
        if scale_factor <= 0.0:
            return
        scale_factor = min(scale_factor, 1.0)
        scaling_new = self.scaling_inverse_activation(self.get_scaling * scale_factor)
        optimizable_tensors = self.replace_tensor_to_optimizer(scaling_new, "scaling")
        self._scaling = optimizable_tensors["scaling"]

    def union_ply(self, plydata, mask_area=None):
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
        if len(binding_names) > 0:
            binding_names = sorted(binding_names, key = lambda x: int(x.split('_')[-1]))
            binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
            for idx, attr_name in enumerate(binding_names):
                binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        # print("debug binding:", binding[:,0].shape, self.binding.shape)
        a_xyz = self._xyz.cpu().detach().numpy()
        a_features_dc = self._features_dc.transpose(1,2).cpu().detach().numpy()
        a_features_extra = self._features_rest.transpose(1,2).cpu().detach().numpy()
        a_opacities = self._opacity.cpu().detach().numpy()
        a_scales = self._scaling.cpu().detach().numpy()
        a_rots = self._rotation.cpu().detach().numpy()
        a_binding = self.binding.cpu().detach().numpy()[..., np.newaxis]

        if mask_area is not None:
            mask = torch.isin(torch.from_numpy(binding[:,0]), self.flame_model.mask.get_fid_by_region([mask_area]).cpu())
            # print (mask.shape)
            mask = mask.cpu()
            # print("debug mask", mask.shape, xyz.shape, xyz[mask].shape, binding.shape)
            xyz = xyz[mask]
            features_dc = features_dc[mask]
            features_extra = features_extra[mask]
            opacities = opacities[mask]
            scales = scales[mask]
            rots = rots[mask]
            binding = binding[mask]

            mask2 = torch.isin(torch.from_numpy(a_binding[:,0]), self.flame_model.mask.get_fid_by_region([mask_area]).cpu(),invert = True)
            a_xyz = a_xyz[mask2]
            a_features_dc = a_features_dc[mask2]
            a_features_extra = a_features_extra[mask2]
            a_opacities = a_opacities[mask2]
            a_scales = a_scales[mask2]
            a_rots = a_rots[mask2]
            a_binding = a_binding[mask2]

        # print("debug feat dc", features_dc.shape)
        a_xyz = np.append(a_xyz, xyz, axis = 0)
        a_features_dc = np.append(a_features_dc, features_dc, axis = 0)
        a_features_extra = np.append(a_features_extra, features_extra, axis = 0)
        a_opacities = np.append(a_opacities, opacities, axis = 0)
        a_scales = np.append(a_scales, scales, axis = 0)
        a_rots = np.append(a_rots, rots, axis = 0)
        a_binding = np.append(a_binding, binding, axis = 0)

        self._xyz = nn.Parameter(torch.tensor(a_xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(a_features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(a_features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(a_opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(a_scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(a_rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.binding = torch.tensor(a_binding, dtype=torch.int32, device="cuda").squeeze(-1)

    def load_ply(self, path, **kwargs):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        try:
            normal_offset = np.asarray(plydata.elements[0]["normal_offset"])
            print("Have normal offset =", normal_offset.shape)
        except Exception as e:
            normal_offset = np.zeros_like(opacities)
            print("Don't have normal offset")

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])


        # LM3D : ------------------------------------------------------------------------------------
        # optional fields
        binding_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("binding")]
        if len(binding_names) > 0:
            binding_names = sorted(binding_names, key = lambda x: int(x.split('_')[-1]))
            binding = np.zeros((xyz.shape[0], len(binding_names)), dtype=np.int32)
            for idx, attr_name in enumerate(binding_names):
                binding[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # self.binding = torch.tensor(binding, dtype=torch.int32, device="cuda").squeeze(-1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        # self.normal_offset = nn.Parameter(torch.tensor(torch.zeros_like(self._xyz[:,0]), dtype=torch.float, device="cuda").requires_grad_(True))
        self.normal_offset = nn.Parameter(torch.tensor(normal_offset, dtype=torch.float, device="cuda").requires_grad_(True))
            
        self.active_sh_degree = self.max_sh_degree

        if len(binding_names) > 0:
            self.binding = torch.tensor(binding, dtype=torch.int32, device="cuda").squeeze(-1)
            
        # print("debug extra_path", 'extra_path' in kwargs and kwargs['extra_path'] is not None)
        # if 'extra_path' in kwargs and kwargs['extra_path'] is not None:
        #     extra_plydata = PlyData.read(kwargs['extra_path'])
        #     self.union_ply(extra_plydata)

        # print("debug teeth_path", 'teeth_path' in kwargs and kwargs['teeth_path'] is not None)
        # if 'teeth_path' in kwargs and kwargs['teeth_path'] is not None:
        #     teeth_plydata = PlyData.read(kwargs['teeth_path'])
        #     self.union_ply(teeth_plydata, 'teeth')
            
        # print("debug eye_path", 'eye_path' in kwargs and kwargs['eye_path'] is not None)
        # if 'eye_path' in kwargs and kwargs['eye_path'] is not None:
        #     eye_plydata = PlyData.read(kwargs['eye_path'])
        #     self.union_ply(eye_plydata, 'eyes')

        ####
        print("debug train", 'is_training' in kwargs and kwargs['is_training'] is True)
        if 'is_training' in kwargs and kwargs['is_training'] is True:
            self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
            num_pts = self._xyz.shape[0]
            fused_color = torch.tensor(np.random.random((num_pts, 3)) / 255.0).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))

        self.binding_counter = torch.bincount(self.binding)

    def replace_tensor_to_optimizer(self, tensor, name):
        ####
        # print("debug tensor:",tensor.shape)
        ####
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if len(group["params"]) != 1 or group["params"][0].shape[0] != mask.shape[0]:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        if self.binding is not None:
            # make sure each face is bound to at least one point after pruning
            binding_to_prune = self.binding[mask]
            counter_prune = torch.zeros_like(self.binding_counter)
            counter_prune.scatter_add_(0, binding_to_prune, torch.ones_like(binding_to_prune, dtype=torch.int32, device="cuda"))
            mask_redundant = (self.binding_counter - counter_prune) > 0
            mask[mask.clone()] = mask_redundant[binding_to_prune]
            print("pruned points:", mask.sum().item(), "out of", mask.shape[0])

        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.normal_offset = optimizable_tensors["normal_offset"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            self.binding_counter.scatter_add_(0, self.binding[mask], -torch.ones_like(self.binding[mask], dtype=torch.int32, device="cuda"))
            self.binding = self.binding[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # rule out parameters that are not properties of gaussians
            if group["name"] not in tensors_dict:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal_offset):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "normal_offset": new_normal_offset
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.normal_offset = optimizable_tensors["normal_offset"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        if self.binding is not None:
            selected_scaling = self.get_scaling[selected_pts_mask]
            face_scaling = self.face_scaling[self.binding[selected_pts_mask]]
            new_scaling = self.scaling_inverse_activation((selected_scaling / face_scaling).repeat(N,1) / (0.8*N))
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_normal_offset = self.normal_offset[selected_pts_mask].repeat(N)
        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask].repeat(N)
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_normal_offset)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, visibility_filter=None, clone_invisible_points: bool = False):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        # LM3D : clone also the invisible points
        print("Clone visible points:", selected_pts_mask.sum().item(), "out of", selected_pts_mask.shape[0])
        if clone_invisible_points and visibility_filter is not None:
            invisibility_filter = torch.logical_not(visibility_filter)
            selected_pts_mask = torch.logical_or(selected_pts_mask, invisibility_filter)
            print("Clone invisible points:", invisibility_filter.sum().item(), "out of", invisibility_filter.shape[0])
            print("Total points cloned:", selected_pts_mask.sum().item(), "out of", selected_pts_mask.shape[0])
        # -----------------------------------------------------------------------------
        
        # new_xyz = torch.tensor(np.random.random((selected_pts_mask.sum().item(), 3)), device="cuda").float()
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_normal_offset = self.normal_offset[selected_pts_mask]
        if self.binding is not None:
            # Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual property and proprietary rights in and to the following code lines and related documentation. Any commercial use, reproduction, disclosure or distribution of these code lines and related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
            new_binding = self.binding[selected_pts_mask].to(dtype=torch.int64) # LM3D : to int64
            self.binding = torch.cat((self.binding, new_binding))
            self.binding_counter.scatter_add_(0, new_binding, torch.ones_like(new_binding, dtype=torch.int32, device="cuda"))
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_normal_offset)

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        visibility_filter=None,
        use_shortersplatting: bool = False,
        scale_prune_quantile: float = 0.99,
        scale_prune_factor: float = 10.0,
        prune_max_fraction: float = 0.05,
        clone_invisible_points: bool = False,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, visibility_filter, clone_invisible_points=clone_invisible_points)
        self.densify_and_split(grads, max_grad, extent)

        # Prune masks
        # scaling_mask = ((self.get_scaling < 0.0010).sum(dim=1) >= 1).squeeze()
        # print("SCALING MASK SUM",scaling_mask.sum())

        opacity_mask = (self.get_opacity < min_opacity).squeeze()

        prune_mask = opacity_mask

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        if use_shortersplatting and prune_mask.shape[0] > 0:
            scale_sum = self.get_scaling.sum(dim=1).float()
            q = min(max(float(scale_prune_quantile), 0.5), 0.9999)
            max_quantile_elems = 1_000_000
            if scale_sum.numel() > max_quantile_elems:
                step = max(1, scale_sum.numel() // max_quantile_elems)
                quantile_source = scale_sum[::step]
            else:
                quantile_source = scale_sum
            scale_threshold = torch.quantile(quantile_source, q) * max(float(scale_prune_factor), 1.0)
            large_scale_mask = scale_sum > scale_threshold

            # Keep regular pruning untouched and only cap additional scale-based pruning.
            extra_mask = torch.logical_and(large_scale_mask, torch.logical_not(prune_mask))
            max_extra = int(prune_mask.shape[0] * min(max(float(prune_max_fraction), 0.0), 1.0))
            if max_extra > 0 and extra_mask.sum().item() > max_extra:
                candidate_idx = torch.nonzero(extra_mask, as_tuple=False).squeeze(1)
                candidate_scores = scale_sum[candidate_idx]
                topk_idx = torch.topk(candidate_scores, k=max_extra, largest=True).indices
                limited_extra_mask = torch.zeros_like(extra_mask)
                limited_extra_mask[candidate_idx[topk_idx]] = True
                extra_mask = limited_extra_mask

            prune_mask = torch.logical_or(prune_mask, extra_mask)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # print("debug viewspace_point_tensor",  viewspace_point_tensor.grad)
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1