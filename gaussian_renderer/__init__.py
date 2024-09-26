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
# from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
# package from this repo: https://github.com/robot0321/diff-gaussian-rasterization-depth-acc/tree/c63d79dc4d59b2965eaf7bada5dda2eae68c08af
from diff_gaussian_rasterization_depth_acc import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization import GaussianRasterizationSettings as GaussianRasterizationSettings2
# from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizer2
from pytorch3d.transforms import quaternion_apply, quaternion_invert, quaternion_multiply

from ..scene.gaussian_model import MultiGaussianModel
from ..utils.sh_utils import eval_sh
from icecream import ic

def render(viewpoint_camera, pc : MultiGaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
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

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_with_sdf(viewpoint_camera, pc : MultiGaussianModel, bg_color : torch.Tensor, objects, \
        scaling_modifier = 1.0, override_color = None, SDF_list = None,\
        debug = False,convert_SHs_python = False,compute_cov3D_python=False):
    """
    Render the scene, with opacity values given by the SDF.
    
    Background tensor (bg_color) must be on GPU!

    Inputs:

    debug: if True, dumps a napshot_fw.dump/snapshot_bw.dump if cuda 
    rasterizer meets an exception
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.gaussians.get_xyz, dtype=pc.gaussians.get_xyz.dtype, \
        requires_grad=True, device="cuda") + 0
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
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.gaussians.get_xyz
    means2D = screenspace_points

    # Opacity now uses the one processed by sdf
    GS_opacity = pc.gaussians.get_opacity

    # opacities = torch.ones((means3D.shape[0], 1), dtype=torch.float, device="cuda")
    # assert len(opacities) == len(pc.masks)
    opacities = torch.ones((means3D.shape[0], 1), dtype=torch.float, device="cuda")
    assert len(opacities) == len(pc.gaussians._masks)

    for obj in objects:
        if obj.sdf is not None:
            obj.sdf.train()
            # normalize and get sdf values
            # pos_sdf_frame = torch.mm(obj.R.t(), (means3D[pc.gaussians._masks == obj.ID,:] - \
            #     obj.pos).t()).t()*obj.scale_tensor
            # ic(pos_sdf_frame)
            # sdfs, normals = obj.sdf.forward_torch(pos_sdf_frame)

            # normalize and get sdf values
            pos_sdf_frame = quaternion_apply(quaternion_invert(obj.rot), \
                means3D[pc.gaussians._masks == obj.ID,:] - obj.pos)*obj.scale_tensor
            sdfs, normals = obj.sdf.forward_torch(pos_sdf_frame)
            sdfs = sdfs/obj.scale_tensor

            obj_opacities = pc.gamma*torch.exp(-pc.beta*sdfs)/torch.pow(1 + torch.exp(-pc.beta*sdfs),2)
            obj.sdf.eval()   
            opacities[pc.gaussians._masks == obj.ID] = obj_opacities   
        else:
            opacities[pc.gaussians._masks == obj.ID] = GS_opacity[pc.gaussians._masks == obj.ID]


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_python:
        cov3D_precomp = pc.gaussians.get_covariance(scaling_modifier)
    else:
        scales = pc.gaussians.get_scaling
        rotations = pc.gaussians.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.gaussians.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, acc, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacities,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_with_sap(viewpoint_camera, pc : MultiGaussianModel, bg_color : torch.Tensor, objects, \
        scaling_modifier = 1.0, override_color = None, SDF_list = None,\
        debug = False,convert_SHs_python = False,compute_cov3D_python=False):
    """
    Render the scene, with opacity values given by the SDF.
    
    Background tensor (bg_color) must be on GPU!

    Inputs:

    debug: if True, dumps a napshot_fw.dump/snapshot_bw.dump if cuda 
    rasterizer meets an exception
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.gaussians.get_xyz, dtype=pc.gaussians.get_xyz.dtype, \
        requires_grad=True, device="cuda") + 0
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
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.gaussians.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.gaussians.get_xyz
    means2D = screenspace_points
    opacity = pc.gaussians.get_opacity



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if compute_cov3D_python:
        cov3D_precomp = pc.gaussians.get_covariance(scaling_modifier)
    else:
        scales = pc.gaussians.get_scaling
        rotations = pc.gaussians.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.gaussians.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, depth, acc, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
