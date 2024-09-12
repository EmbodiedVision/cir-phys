#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from lietorch import SE3, Sim3
from ev_sdf_utils import grid_interp


def grid_grads(vol, fill_value=torch.nan):
    """
    Finite-difference gradients for a given volume
    The input volume can be batched.
    """
    grads = torch.stack([
        torch.cat([
            vol.new_full(vol.shape[:-3] + (1,) + vol.shape[-2:], fill_value),
            (vol[..., 2:, :, :] - vol[..., :-2, :, :]) / 2.,
            vol.new_full(vol.shape[:-3] + (1,) + vol.shape[-2:], fill_value)
        ], dim=-3),
        torch.cat([
            vol.new_full(vol.shape[:-2] + (1,) + vol.shape[-1:], fill_value),
            (vol[..., :, 2:, :] - vol[..., :, :-2, :]) / 2.,
            vol.new_full(vol.shape[:-2] + (1,) + vol.shape[-1:], fill_value)
        ], dim=-2),
        torch.cat([
            vol.new_full(vol.shape[:-1] + (1,), fill_value),
            (vol[..., :, :, 2:] - vol[..., :, :, :-2]) / 2.,
            vol.new_full(vol.shape[:-1] + (1,), fill_value)
        ], dim=-1)
    ], dim=0 if vol.ndim == 3 else 1)

    return grads


def _query_sdf(pc_obj, sdf_volume, sdf_offset, sdf_scale, jacobian=False, sdf_grads=None):
    pc_normalized = (pc_obj - sdf_offset.unsqueeze(1)) / sdf_scale[:, None, None]

    pc_idxs = (pc_normalized + 1.) / 2. * sdf_volume.shape[-1]

    sdfs = grid_interp(sdf_volume, pc_idxs, bounds_error=False) * sdf_scale.unsqueeze(1)
    valid = ~sdfs.isnan()

    if jacobian:
        grads = grid_interp(sdf_grads, pc_idxs, bounds_error=False) * sdf_volume.shape[-1] / 2.
        valid &= ~grads.isnan().any(dim=-1)
        return sdfs, grads, valid

    return sdfs, valid


def intersection_residuals(poses, pcis, ii, jj, sdf_volumes, sdf_offsets, sdf_scales, jacobian=False, sdf_grads=None):
    pcs_j = relative_pose_transform(poses, pcis, ii, jj, jacobian=False)

    sdf_qj = _query_sdf(pcs_j, sdf_volumes[jj], sdf_offsets[jj], sdf_scales[jj], jacobian=jacobian,
                        sdf_grads=sdf_grads[jj])
    sdf_qi = _query_sdf(pcis[ii], sdf_volumes[ii], sdf_offsets[ii], sdf_scales[ii], jacobian=False,
                        sdf_grads=sdf_grads[ii])
    valid = sdf_qj[-1] & sdf_qi[-1]
    r = sdf_qj[0] + sdf_qi[0]
    valid &= r < 0

    r[~valid] = 0.

    if jacobian:
        Jis = pcs_j.new_zeros(pcs_j.shape[0], pcs_j.shape[1], poses.manifold_dim)
        if valid.any():
            Jr = sdf_qj[1][valid]
            Ji = relative_pose_transform_jacobian(poses[torch.repeat_interleave(jj, valid.sum(dim=1))], pcs_j[valid])
            Ji = (Jr.unsqueeze(-2) @ Ji).squeeze(-2)
            Jis[valid] = Ji

    if jacobian:
        return r, Jis, valid
    return r, valid


def relative_pose_transform_jacobian(Gjs, Xs):
    X, Y, Z = Xs.unbind(dim=-1)

    if X.ndim == 2:
        B, N = X.shape
    else:
        B, N = 1, X.shape[0]
    o = torch.zeros_like(Z)
    i = torch.ones_like(Z)
    # action jacobian (Ja)
    if isinstance(Gjs, SE3):
        Ja = torch.stack([
            i, o, o, o, Z, -Y,
            o, i, o, -Z, o, X,
            o, o, i, Y, -X, o,
        ], dim=-1).view(B, N, 3, 6)

    elif isinstance(Gjs, Sim3):
        Ja = torch.stack([
            i, o, o, o, Z, -Y, X,
            o, i, o, -Z, o, X, Y,
            o, o, i, Y, -X, o, Z,
        ], dim=-1).view(B, N, 3, 7)

    if X.ndim == 2:
        Ji = Gjs[:, None, None].inv().adjT(Ja)
    else:
        Ji = Gjs[None, :, None].inv().adjT(Ja).squeeze(0)
    # Jj = -Ji

    return Ji  # , Jj


def relative_pose_transform(poses, pcs, ii, jj, jacobian=False):
    Gij = poses[jj].inv() * poses[ii]

    pcs_j = Gij[:, None] * pcs[ii]

    if jacobian:
        Ji, Jj = relative_pose_transform_jacobian(poses[jj], pcs_j)
        return pcs_j, (Ji, Jj)

    return pcs_j


def generate_pcis(shape, sdf_scales, sdf_offsets, sampling=False, n_samples=100000, generator=None):
    if sampling:
        shape = torch.tensor(shape, device=sdf_scales.device)
        grid = torch.rand((len(sdf_scales), n_samples, 3), generator=generator).to(device=sdf_scales.device) * (shape - 1) / shape * 2. - 1.
    else:
        # Using linspace to avoid computations with arange
        # np.linspace(-1., 1., sdf1_shape[0]+1)[:-1] == 2 * np.arange(sdf.shape[0]) / sdf.shape[0] - 1.
        grid = torch.stack(torch.meshgrid(torch.linspace(-1., 1., shape[0] + 1)[:-1],
                                          torch.linspace(-1., 1., shape[1] + 1)[:-1],
                                          torch.linspace(-1., 1., shape[2] + 1)[:-1], indexing='ij'),
                           dim=-1).reshape(1, -1, 3).repeat(len(sdf_scales), 1, 1).to(device=sdf_scales.device)
    pcs = (grid * sdf_scales[:, None, None] + sdf_offsets.unsqueeze(1))

    return pcs
