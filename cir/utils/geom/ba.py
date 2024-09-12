#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ===================================================================
#
# This file originates from the CIR repository at
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/blob/c50df7816714007c7f2f5188995807b3b396ad3d/utils/geom/ba.py,
# licensed under the MIT license (see CIR-LICENSE) and was modified in the places marked in this file.
#
import gin
import lietorch
import torch
import trimesh
from ev_sdf_utils import marching_cubes
from torch_scatter import scatter_sum

from .chol import schur_solve
from .intersection_ops import intersection_residuals
from .projective_ops_rgb import \
    projective_transform as projective_transform_rgb
from .projective_ops_rgbd import \
    projective_transform as projective_transform_rgbd

import matplotlib.pyplot as plt


# --------------
# Code block added compared to the original CIR version
# Functions for debug visualizations
def plot_jrs(pcs, Jrs, downsampling=10):
    pcs = pcs[::downsampling].cpu()
    Jrs = Jrs[::downsampling].cpu()

    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(pcs[:, 0], pcs[:, 1], pcs[:, 2], Jrs[:, 0], Jrs[:, 1], Jrs[:, 2], normalize=True, length=0.001)
    plt.show()
# --------------



# --------------
# Code block added compared to the original CIR version
def show_meshes_points(sdf_volumes, sdf_offsets, sdf_scales, i, j, poses, pcs_j):
    scene = trimesh.Scene()

    vi, fi = marching_cubes(sdf_volumes[i], 0.)
    vi = (vi / vi.new_tensor(sdf_volumes[i].shape) * 2. - 1.) * sdf_scales[i] + sdf_offsets[i]

    mi = trimesh.Trimesh(vi.cpu(), fi.cpu(), vertex_colors=[1., 0., 0., 0.4])

    vj, fj = marching_cubes(sdf_volumes[j], 0.)
    vj = (vj / vj.new_tensor(sdf_volumes[j].shape) * 2. - 1.) * sdf_scales[j] + sdf_offsets[j]

    mj = trimesh.Trimesh(vj.cpu(), fj.cpu(), vertex_colors=[0., 1., 0., 0.4])

    pc = trimesh.PointCloud(pcs_j.cpu(), colors=[0., 0., 1., 0.4])

    scene.add_geometry(mj)
    scene.add_geometry(mi, transform=(poses[j].inv() * poses[i]).matrix().cpu())
    scene.add_geometry(pc)

    scene.show()
# --------------

"""
Modified BD-PnP Solver
"""

# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:, v], ii[v] * m + jj[v], dim=1, dim_size=n * m)


def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:, v], ii[v], dim=1, dim_size=n)


# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])


# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


@gin.configurable()
def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, motion_only=False):
    """ Full Bundle Adjustment """

    fixedp = fixedd = (poses.shape[1] - 1)
    M = 1

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = projective_transform_rgb(
        poses, disps, intrinsics, ii, jj, jacobian=True, return_depth=False)
    target = target[..., :2]
    weight = weight[..., :2]

    r = (target - coords).view(B, N, -1, 1)
    w_scale = 0.001
    w = w_scale * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2, 3)
    wJjT = (w * Jj).transpose(2, 3)

    Jz = Jz.reshape(B, N, ht * wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)
    Ej = (wJjT.view(B, N, D, ht * wd, -1) * Jz[:, :, None]).sum(dim=-1)

    w = w.view(B, N, ht * wd, -1)
    r = r.view(B, N, ht * wd, -1)
    wk = torch.sum(w * r * Jz, dim=-1)
    Ck = torch.sum(w * Jz * Jz, dim=-1)

    kk = ii.clone()
    kk = kk - fixedd

    # only optimize keyframe poses
    P = 1
    ii = ii - fixedp
    jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7 # This is C from Ceres pdf: http://ceres-solver.org/nnls_solving.html?highlight=normal%20equations#equation-hblock

    H = H.view(B, P, P, D, D)  # This is B from Ceres pdf.
    E = E.view(B, P, M, D, ht * wd)

    ### 3: solve the system ###
    if motion_only:
        E = torch.zeros_like(E)
    dx, dz = schur_solve(H, E, C, v, w, ep=0.1, lm=0.0001)
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B, -1, ht, wd), torch.arange(M) + fixedd)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)
    return poses, disps


"""
BD-PnP Solver
"""


class DenseSystemSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # --------------
        # Code block modified compared to the original CIR version
        U, _ = torch.linalg.cholesky_ex(H)
        # --------------

        xs = torch.cholesky_solve(b, U)
        ctx.save_for_backward(U, xs)
        return xs

    @staticmethod
    def backward(ctx, grad_x):
        grad_x = torch.where(torch.isnan(grad_x),
                             torch.zeros_like(grad_x), grad_x)

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1, -2))

        return dH, dz


def _linearize_moba(target, weight, poses, depths, intrinsics, ii, jj):
    bdim, mdim = B, M = poses.shape[:2]
    ddim = D = poses.manifold_dim
    ndim = N = ii.shape[0]
    ### 1: commpute jacobians and residuals ###
    # Ji = dcoords/dGi (and similar for Jj)
    coords, val, (Ji, Jj) = projective_transform_rgbd(
        poses, depths, intrinsics, ii, jj, jacobian=True)
    val = val * (depths[:, ii] > 0.1).float()
    val = val.unsqueeze(-1)

    # dr/dGi = dr/dcoords Ji = -Ji (!!!) ==> That's why there's no minus for dx!
    r = (target - coords).view(B, N, -1, 1)
    w = (val * weight).view(B, N, -1, 1)
    ### 2: construct linear system ###
    Ji = Ji.view(B, N, -1, 6)
    Jj = Jj.view(B, N, -1, 6)
    wJiT = (.001 * w * Ji).transpose(2, 3)
    wJjT = (.001 * w * Jj).transpose(2, 3)
    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)
    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    device = Jj.device

    H = torch.zeros(bdim, mdim * mdim, ddim, ddim, device=device)
    v = torch.zeros(bdim, mdim, ddim, device=device)

    H.scatter_add_(1, (ii * mdim + ii).view(1, ndim, 1, 1).repeat(bdim, 1, ddim, ddim), Hii)
    H.scatter_add_(1, (ii * mdim + jj).view(1, ndim, 1, 1).repeat(bdim, 1, ddim, ddim), Hij)
    H.scatter_add_(1, (jj * mdim + ii).view(1, ndim, 1, 1).repeat(bdim, 1, ddim, ddim), Hji)
    H.scatter_add_(1, (jj * mdim + jj).view(1, ndim, 1, 1).repeat(bdim, 1, ddim, ddim), Hjj)
    H = H.view(bdim, mdim, mdim, ddim, ddim)

    v.scatter_add_(1, ii.view(1, ndim, 1).repeat(bdim, 1, ddim), vi)
    v.scatter_add_(1, jj.view(1, ndim, 1).repeat(bdim, 1, ddim), vj)

    return H, v

# --------------
# Code block added compared to the original CIR version
def _linearize_inter(pcis, poses, sdfs):
    M = poses.shape[0]
    D = poses.manifold_dim

    sdf_volumes = sdfs['volumes']
    sdf_grads = sdfs['grads']
    sdf_offsets = sdfs['offsets']
    sdf_scales = sdfs['scales']

    ii = torch.arange(M, device=poses.device).repeat_interleave(M - 1)
    jj = torch.arange(M, device=poses.device).repeat(M)[(1 - torch.eye(M)).bool().reshape(-1)]

    N = ii.shape[0]

    rs, Ji, valid = intersection_residuals(poses, pcis, ii, jj, sdf_volumes, sdf_offsets, sdf_scales, jacobian=True,
                                           sdf_grads=sdf_grads)
    # Jj = -Ji

    w = valid.float()

    valid_inds = torch.where(valid)
    indices = torch.stack([
        valid_inds[0].repeat_interleave(D),
        valid_inds[1].repeat_interleave(D),
        torch.arange(D, device=valid.device).repeat(len(valid_inds[0]))
    ])
    wJiT = (1 * w.view(N, -1, 1) * torch.sparse_coo_tensor(indices, Ji[valid].reshape(-1), Ji.shape)).transpose(-1, -2)

    Hii = torch.stack([torch.sparse.mm(jit, ji) for jit, ji in zip(wJiT, Ji)])
    # Hij = -Hii
    # Hji = Hij
    # Hjj = Hii
    vi = (wJiT.to_dense() * rs.unsqueeze(1)).sum(dim=-1)
    # vj = -vi

    device = Ji.device

    H = torch.zeros(M * M, D, D, device=device)
    v = torch.zeros(M, D, device=device)

    H.scatter_add_(0, (ii * M + ii).view(N, 1, 1).repeat(1, D, D), Hii)
    H.scatter_add_(0, (ii * M + jj).view(N, 1, 1).repeat(1, D, D), -Hii)
    H.scatter_add_(0, (jj * M + ii).view(N, 1, 1).repeat(1, D, D), -Hii)
    H.scatter_add_(0, (jj * M + jj).view(N, 1, 1).repeat(1, D, D), Hii)

    H = H.view(M, M, D, D)

    v.scatter_add_(0, ii.view(N, 1).repeat(1, D), vi)
    v.scatter_add_(0, jj.view(N, 1).repeat(1, D), -vi)

    return H, v
# --------------


# --------------
# Code block modified compared to the original CIR version: add arguments for intersection constraint
def _step_moba(target, weight, poses, depths, intrinsics, ii, jj, intersection=False, pcis=None, sdfs=None, G_bg=None,
               ep_lmbda=10.0, lm_lmbda=0.00001):
# --------------
    bd, kd, ht, wd = depths.shape
    md = poses.shape[1]
    D = poses.manifold_dim

    H, v = _linearize_moba(target, weight, poses, depths, intrinsics, ii, jj)

    H = H.permute(0, 1, 3, 2, 4).reshape(bd, D * md, D * md)
    v = v.reshape(bd, D * md, 1)

    dI = torch.eye(D * md, device=H.device)
    _H = H + ep_lmbda * dI + lm_lmbda * H * dI

    # fix first pose
    _H = _H[:, -D:, -D:]
    _v = v[:, -D:]

    # --------------
    # Code block added compared to the original CIR version
    # Only compute intersection if we actually have multiple objects
    if intersection and (G_bg is not None or poses.shape[0] > 1):
        H_inter, v_inter = _linearize_inter(
            pcis, lietorch.cat([G_bg, poses[:, -1]], dim=0) if G_bg else poses[:, -1], sdfs)

        bd_inter = bd + 1 if G_bg else bd
        H_inter = H_inter.permute(0, 2, 1, 3).reshape(D * bd_inter, D * bd_inter)
        v_inter = v_inter.reshape(bd_inter * D, 1)

        if G_bg:
            # Fix background position
            H_inter = H_inter[D:, D:]
            v_inter = v_inter[D:, :]

        _H = torch.block_diag(*_H) + H_inter
        # As H_inter and v_inter are built on +dr_inter/dG, we need a minus here to be consistent with the previous
        # MoBA step
        _v = _v.reshape(bd * D, 1) - v_inter
    # --------------

    dx = DenseSystemSolver.apply(_H, _v)
    dx = dx.view(bd, 1, D)
    # dx = dx.clamp(-2.0, 2.0)

    fill = torch.zeros_like(dx[:, :1].repeat(1, md - 1, 1))
    dx = torch.cat([fill, dx], dim=1)

    poses = poses.retr(dx)

    return poses, depths, intrinsics

# --------------
# Code block modified compared to the original CIR version: add arguments for intersection constraint
def MoBA(target, weight, poses, depths, intrinsics, num_steps, ii, jj, intersection=False, pcis=None, sdfs=None, G_bg=None):
    """ Motion only bundle adjustment """
    for itr in range(num_steps):
        poses, depths, intrinsics = _step_moba(target, weight, poses, depths, intrinsics, ii, jj,
                                               intersection=intersection, pcis=pcis, sdfs=sdfs, G_bg=G_bg)
    return poses, depths, intrinsics
# --------------
