#
# Copyright (c) 2024 Max Planck Society
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import trimesh
from ev_sdf_utils import grid_interp, marching_cubes

from bop_toolkit_lib import inout, dataset_params, config, misc

DEBUG = True

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
    # Top N pose estimates (with the highest score) to be evaluated for each
    # object class in each image.
    # Options: 0 = all, -1 = given by the number of GT poses.
    'n_top': 1,

    # Pose error function.
    # Options: 'vsd', 'mssd', 'mspd', 'ad', 'adi', 'add', 'cus', 're', 'te, etc.
    'error_type': 'vsd',

    # VSD parameters.
    'vsd_deltas': {
        'hb': 15,
        'icbin': 15,
        'icmi': 15,
        'itodd': 5,
        'lm': 15,
        'lmo': 15,
        'ruapc': 15,
        'tless': 15,
        'tudl': 15,
        'tyol': 15,
        'ycbv': 15,
        'hope': 15,
    },
    'vsd_taus': list(np.arange(0.05, 0.51, 0.05)),
    'vsd_normalized_by_diameter': True,

    # MSSD/MSPD parameters (see misc.get_symmetry_transformations).
    'max_sym_disc_step': 0.01,

    # Whether to ignore/break if some errors are missing.
    'skip_missing': True,

    # Type of the renderer (used for the VSD pose error function).
    'renderer_type': 'vispy',  # Options: 'vispy', 'cpp', 'python'.

    # Names of files with results for which to calculate the errors (assumed to be
    # stored in folder p['results_path']). See docs/bop_challenge_2019.md for a
    # description of the format. Example results can be found at:
    # https://bop.felk.cvut.cz/media/data/bop_sample_results/bop_challenge_2019_sample_results.zip
    'result_filenames': [
        '/path/to/csv/with/results',
    ],

    # Folder with results to be evaluated.
    'results_path': config.results_path,

    # Folder for the calculated pose errors and performance scores.
    'eval_path': config.eval_path,

    # Folder containing the BOP datasets.
    'datasets_path': config.datasets_path,

    # File with a list of estimation targets to consider. The file is assumed to
    # be stored in the dataset folder.
    'targets_filename': 'test_targets_bop19.json',

    # Template of path to the output file with calculated errors.
    'out_errors_tpath': os.path.join(
        '{eval_path}', '{result_name}', '{error_sign}',
        'errors_{scene_id:06d}.json')
}


def vis_meshes(Rs, ts, obj_ids, sdfs):
    scene = trimesh.Scene()
    for R, t, obj_id in zip(Rs, ts, obj_ids):
        v, f = marching_cubes(sdfs[obj_id]['sdf'], 0.)
        v = (v / sdfs[obj_id]['sdf'].shape[0] - 0.5) * 2. * sdfs[obj_id]['scale'] + sdfs[obj_id]['offset']
        m = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy())
        transform = torch.eye(4)
        transform[:3, :3] = R
        transform[:3, 3:] = t
        scene.add_geometry(m, transform=transform.cpu().numpy())

    scene.show()


def pen(Rs, ts, obj_ids, sdfs):
    """Penetration of objects in given poses.

    :param Rs: list of 3x3 ndarrays or nx3x3 ndarray with rotation matrices.
    :param ts: list of 3x1 ndarrays or nx3x1 ndarray with translation vectors.
    :param obj_ids: list of object ids corresponding to the poses.
    :param sdfs: dict mapping obj_ids to sdf grids.
    """
    pairwise_penetrations = torch.zeros((len(Rs), len(Rs)), device=DEVICE)
    rel_pairwise_penetrations = torch.zeros((len(Rs), len(Rs)), device=DEVICE)
    penetrations = []
    rel_penetrations = []
    for i, (R1, t1, id1) in enumerate(zip(Rs, ts, obj_ids)):
        sdf1 = sdfs[id1]
        sdf1_shape = torch.tensor(sdf1['sdf'].shape, device=DEVICE)

        # Using linspace to avoid computations with arange
        # np.linspace(-1., 1., sdf1_shape[0]+1)[:-1] == 2 * np.arange(sdf.shape[0]) / sdf.shape[0] - 1.
        grid1 = torch.stack(torch.meshgrid(torch.linspace(-1., 1., sdf1_shape[0] + 1)[:-1],
                                           torch.linspace(-1., 1., sdf1_shape[1] + 1)[:-1],
                                           torch.linspace(-1., 1., sdf1_shape[2] + 1)[:-1], indexing='ij'),
                            dim=-1).reshape(-1, 3).to(DEVICE)
        pc1 = grid1 * sdf1['scale'] + sdf1['offset']
        pc1_w = R1 @ pc1.T + t1

        vol1 = (sdf1['sdf'] < 0.).sum() * (2 * sdf1['scale'] / sdf1_shape).prod()

        pen_mask = torch.zeros(torch.Size(sdf1_shape), dtype=bool, device=DEVICE)
        for j, (R2, t2, id2) in enumerate(zip(Rs, ts, obj_ids)):
            if i == j:
                continue
            sdf2 = sdfs[id2]
            sdf2_shape = torch.tensor(sdf2['sdf'].shape, device=DEVICE)

            pc1_2 = R2.T @ (pc1_w - t2)
            pc2 = (pc1_2.T - sdf2['offset']) / sdf2['scale']

            inds2 = (pc2 + 1.) / 2. * sdf2_shape
            sdfs2 = grid_interp(sdf2['sdf'], inds2.to(sdf2['sdf'].dtype), bounds_error=False)

            mask = ~torch.isnan(sdfs2)
            sdfs1_masked = sdf1['sdf'].reshape(-1)[mask]
            sdfs2_masked = sdfs2[mask]

            # This is approximately the penetration volume, i.e. number of voxels where we are "inside" in
            # both objects times the size of a single voxel in mm^3.
            pairwise_penetrations[i, j] = (((sdfs1_masked < 0) & (sdfs2_masked < 0)).sum()
                                           * (2 * sdf1['scale'] / sdf1_shape).prod())
            rel_pairwise_penetrations[i, j] = pairwise_penetrations[i, j] / vol1

            pen_mask[mask.reshape(pen_mask.shape)] |= (sdfs1_masked < 0) & (sdfs2_masked < 0)

        penetrations.append(pen_mask.sum() * (2 * sdf1['scale'] / sdf1_shape).prod())
        rel_penetrations.append(pen_mask.sum() * (2 * sdf1['scale'] / sdf1_shape).prod() / vol1)

    return penetrations, rel_penetrations, pairwise_penetrations, rel_pairwise_penetrations


def main():
    parser = ArgumentParser()
    parser.add_argument('--result_filenames',
                        default=','.join(p['result_filenames']),
                        help='Comma-separated names of files with results.')
    parser.add_argument('--results_path', default=p['results_path'])
    parser.add_argument('--eval_path', default=p['eval_path'])
    parser.add_argument('--targets_filename', default=p['targets_filename'])
    parser.add_argument('--eval_bg_pen', action='store_true')

    args = parser.parse_args()

    p['result_filenames'] = args.result_filenames.split(',')
    p['eval_path'] = str(args.eval_path)
    p['results_path'] = str(args.results_path)
    p['targets_filename'] = str(args.targets_filename)

    for result_filename in p['result_filenames']:
        misc.log('===========')
        misc.log('EVALUATING: {}'.format(result_filename))
        misc.log('===========')

        # Parse info about the method and the dataset from the filename.
        result_name = os.path.splitext(os.path.basename(result_filename))[0]
        result_info = result_name.split('_')
        method = str(result_info[0])
        dataset_info = result_info[1].split('-')
        dataset = str(dataset_info[0])
        split = str(dataset_info[1])
        split_type = '_'.join(dataset_info[2:]) if len(dataset_info) > 2 else None
        split_type_str = ' - ' + split_type if split_type is not None else ''

        # Load dataset parameters.
        dp_split = dataset_params.get_split_params(
            p['datasets_path'], dataset, split, split_type)

        # Load the estimation targets.
        targets = inout.load_json(
            os.path.join(dp_split['base_path'], p['targets_filename']))
        # Organize the targets by scene, image and object.
        misc.log('Organizing estimation targets...')
        targets_org = {}
        for target in targets:
            targets_org.setdefault(target['scene_id'], {}).setdefault(
                target['im_id'], {})[target['obj_id']] = target

        poses = inout.load_bop_results(os.path.join(p['results_path'], result_filename))
        for i in range(len(poses)):
            poses[i]['R'] = torch.from_numpy(poses[i]['R']).to(DEVICE)
            poses[i]['t'] = torch.from_numpy(poses[i]['t']).to(DEVICE)

        dp_model = dataset_params.get_model_params(
            p['datasets_path'], dataset)
        if dataset == 'lmo':
            # Not all objects are in LMO
            dp_model = dataset_params.get_model_params(
                p['datasets_path'], 'lm'
            )

        sdfs = {}
        misc.log('Loading object sdfs...')
        for obj_id in dp_model['obj_ids']:
            sdfs[obj_id] = inout.load_sdf(dp_model['sdf_tpath'].format(obj_id=obj_id))
            sdfs[obj_id] = {k: torch.from_numpy(v).to(DEVICE) for k, v in sdfs[obj_id].items()}

        if args.eval_bg_pen:
            bg_model_name = None
            if dataset == 'synpick':
                bg_model_name = 'tote.npz'
            elif dataset == 'ycbv':
                bg_model_name = 'plane.npz'
                with open(os.path.join(dp_split['base_path'], 'sporeagent', 'test_posecnn_plane.pkl'), 'rb') as f:
                    samples = pickle.load(f)

            if bg_model_name is not None:
                sdfs['bg'] = inout.load_sdf(os.path.join(os.path.dirname(dp_model['sdf_tpath']), bg_model_name))
                sdfs['bg'] = {k: torch.from_numpy(v).to(DEVICE) for k, v in sdfs['bg'].items()}
                sdfs['bg']['scale'] *= 1000.
                sdfs['bg']['offset'] *= 1000.

        poses = pd.DataFrame(poses)
        os.makedirs(os.path.join(p['eval_path'], os.path.splitext(result_filename)[0], 'error=penetration'), exist_ok=True)
        for scene_id, scene_targets in targets_org.items():
            scene_camera = inout.load_scene_camera(dp_split["scene_camera_tpath"].format(scene_id=scene_id))

            scene_penetrations = []
            scene_poses = poses[poses['scene_id'] == scene_id]
            for im_id, im_targets in scene_targets.items():
                print(f"Computing penetration for scene: {scene_id}, im: {im_id}")
                im_poses = scene_poses[scene_poses['im_id'] == im_id]
                Rs, ts, obj_ids = list(im_poses['R']), list(im_poses['t']), list(im_poses['obj_id'])
                if 'bg' in sdfs and len(im_poses) > 0:
                    if dataset == 'ycbv':
                        matching_samples = [s for s in samples if s['scene'] == scene_id and s['frame'] == im_id]
                        assert len(matching_samples) == 1
                        plane_pose = torch.tensor(matching_samples[0]['plane']).to(DEVICE).to(Rs[0].dtype)
                        R_bg = plane_pose[:3, :3]
                        t_bg = plane_pose[:3, 3:]
                    elif dataset == 'synpick':
                        R_bg = torch.from_numpy(scene_camera[im_id]['cam_R_w2c']).to(DEVICE).to(Rs[0].dtype)
                        t_bg = torch.from_numpy(scene_camera[im_id]['cam_t_w2c']).to(DEVICE).to(Rs[0].dtype)
                    Rs.append(R_bg)
                    ts.append(t_bg)
                    obj_ids.append('bg')

                # vis_meshes(Rs, ts, obj_ids, sdfs)
                penetrations, rel_penetrations, pairwise_penetrations, rel_pairwise_penetrations \
                    = pen(Rs, ts, obj_ids, sdfs)
                for est_id, obj_id in enumerate(im_poses['obj_id']):
                    errs = {
                        'pen': penetrations[est_id].item(),
                        'rel_pen': rel_penetrations[est_id].item(),
                        'pairwise_pen': pairwise_penetrations[est_id].tolist(),
                        'rel_pairwise_pen': rel_pairwise_penetrations[est_id].tolist()
                    }

                    scene_penetrations.append({
                        'errors': errs,
                        'est_id': est_id,
                        'im_id': im_id,
                        'obj_id': obj_id,
                        'score': list(im_poses['score'])[est_id]
                    })
            inout.save_json(
                os.path.join(p['eval_path'], os.path.splitext(result_filename)[0], 'error=penetration', f'errors_{scene_id:06d}.json'),
                scene_penetrations)

if __name__ == "__main__":
    main()
