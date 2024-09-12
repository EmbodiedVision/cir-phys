#
# Copyright (c) 2024 Max Planck Society
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import time

import numpy as np

from bop_toolkit_lib import config, misc, dataset_params, inout

# PARAMETERS (can be overwritten by the command line arguments below).
################################################################################
p = {
  # Threshold of correctness for different pose error functions.
  'correct_th': {
    'vsd': [0.3],
    'mssd': [0.2],
    'mspd': [10],
    'cus': [0.5],
    'rete': [5.0, 5.0],  # [deg, cm].
    're': [5.0],  # [deg].
    'te': [5.0],  # [cm].
    'proj': [5.0],  # [px].
    'ad': [0.1],
    'add': [0.1],
    'adi': [0.1],
    'penetration': [-1],
  },

  # Pose errors that will be normalized by object diameter before thresholding.
  'normalized_by_diameter': ['ad', 'add', 'adi', 'mssd'],

  # Pose errors that will be normalized the image width before thresholding.
  'normalized_by_im_width': ['mspd'],

  # Minimum visible surface fraction of a valid GT pose.
  # -1 == k most visible GT poses will be considered, where k is given by
  # the "inst_count" item loaded from "targets_filename".
  'visib_gt_min': -1,

  # Paths (relative to p['eval_path']) to folders with pose errors calculated
  # using eval_calc_errors.py.
  # Example: 'hodan-iros15_lm-test/error=vsd_ntop=1_delta=15_tau=20_cost=step'
  'error_dir_paths': [
    r'/path/to/calculated/errors',
  ],

  # Folder for the calculated pose errors and performance scores.
  'eval_path': config.eval_path,

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # File with a list of estimation targets to consider. The file is assumed to
  # be stored in the dataset folder.
  'targets_filename': 'test_targets_bop19.json',

  # Template of path to the input file with calculated errors.
  'error_tpath': os.path.join(
    '{eval_path}', '{error_dir_path}', 'errors_{scene_id:06d}.json'),

  # Template of path to the output file with established matches and calculated
  # scores.
  'out_matches_tpath': os.path.join(
    '{eval_path}', '{error_dir_path}', 'matches_{score_sign}.json'),
  'out_scores_tpath': os.path.join(
    '{eval_path}', '{error_dir_path}', 'scores_{score_sign}.json'),
}
################################################################################

# Command line arguments.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

# Define the command line arguments.
for err_type in p['correct_th']:
  parser.add_argument(
    '--correct_th_' + err_type,
    default=','.join(map(str, p['correct_th'][err_type])))

parser.add_argument('--normalized_by_diameter',
                    default=','.join(p['normalized_by_diameter']))
parser.add_argument('--normalized_by_im_width',
                    default=','.join(p['normalized_by_im_width']))
parser.add_argument('--visib_gt_min', default=p['visib_gt_min'])
parser.add_argument('--error_dir_paths', default=','.join(p['error_dir_paths']),
                    help='Comma-sep. paths to errors from eval_calc_errors.py.')
parser.add_argument('--eval_path', default=p['eval_path'])
parser.add_argument('--datasets_path', default=p['datasets_path'])
parser.add_argument('--targets_filename', default=p['targets_filename'])
parser.add_argument('--error_tpath', default=p['error_tpath'])
parser.add_argument('--out_matches_tpath', default=p['out_matches_tpath'])
parser.add_argument('--out_scores_tpath', default=p['out_scores_tpath'])

# Process the command line arguments.
args = parser.parse_args()

for err_type in p['correct_th']:
  p['correct_th'][err_type] =\
    list(map(float, args.__dict__['correct_th_' + err_type].split(',')))

p['normalized_by_diameter'] = args.normalized_by_diameter.split(',')
p['normalized_by_im_width'] = args.normalized_by_im_width.split(',')
p['visib_gt_min'] = float(args.visib_gt_min)
p['error_dir_paths'] = args.error_dir_paths.split(',')
p['eval_path'] = str(args.eval_path)
p['datasets_path'] = str(args.datasets_path)
p['targets_filename'] = str(args.targets_filename)
p['error_tpath'] = str(args.error_tpath)
p['out_matches_tpath'] = str(args.out_matches_tpath)
p['out_scores_tpath'] = str(args.out_scores_tpath)

misc.log('-----------')
misc.log('Parameters:')
for k, v in p.items():
  misc.log('- {}: {}'.format(k, v))
misc.log('-----------')

# Calculation of the performance scores.
# ------------------------------------------------------------------------------
for error_dir_path in p['error_dir_paths']:
    misc.log('Processing: {}'.format(error_dir_path))

    time_start = time.time()

    # Parse info about the errors from the folder name.
    error_sign = os.path.basename(error_dir_path)
    err_type = str(error_sign.split('_')[0].split('=')[1])
    # n_top = int(error_sign.split('_')[1].split('=')[1])
    result_info = os.path.basename(os.path.dirname(error_dir_path)).split('_')
    method = result_info[0]
    dataset_info = result_info[1].split('-')
    dataset = dataset_info[0]
    split = dataset_info[1]
    split_type = '_'.join(dataset_info[2:]) if len(dataset_info) > 2 else None

    # Evaluation signature.
    score_sign = misc.get_score_signature(
        p['correct_th'][err_type], p['visib_gt_min'])

    misc.log('Calculating score - error: {}, method: {}, dataset: {}.'.format(
        err_type, method, dataset))

    # Load dataset parameters.
    dp_split = dataset_params.get_split_params(
    p['datasets_path'], dataset, split, split_type)

    model_type = 'eval'
    dp_model = dataset_params.get_model_params(
    p['datasets_path'], dataset, model_type)

    # Load info about the object models.
    models_info = inout.load_json(dp_model['models_info_path'], keys_to_int=True)

    # Load the estimation targets to consider.
    targets = inout.load_json(
    os.path.join(dp_split['base_path'], p['targets_filename']))
    scene_im_ids = {}

    # Organize the targets by scene, image and object.
    misc.log('Organizing estimation targets...')
    targets_org = {}
    for target in targets:
        targets_org.setdefault(target['scene_id'], {}).setdefault(
            target['im_id'], {})[target['obj_id']] = target

    avg_rel_pens_scene = {}
    avg_abs_pens_scene = {}
    max_rel_pens_scene = {}
    max_abs_pens_scene = {}

    rel_pens_obj = {}
    abs_pens_obj = {}

    avg_rel_pens_im = {}
    avg_abs_pens_im = {}
    max_rel_pens_im = {}
    max_abs_pens_im = {}
    for scene_id, scene_targets in targets_org.items():
        misc.log('Processing scene {} of {}...'.format(scene_id, dataset))

        scene_errs_path = p['error_tpath'].format(
            eval_path=p['eval_path'], error_dir_path=error_dir_path,
            scene_id=scene_id)
        scene_errs = inout.load_json(scene_errs_path, keys_to_int=True)

        rel_pens_scene = [e['errors']['rel_pen'] for e in scene_errs]
        abs_pens_scene = [e['errors']['pen'] for e in scene_errs]

        avg_rel_pens_scene[scene_id] = np.mean(rel_pens_scene) if len(rel_pens_scene) > 0 else np.nan
        avg_abs_pens_scene[scene_id] = np.mean(abs_pens_scene) if len(abs_pens_scene) > 0 else np.nan
        max_rel_pens_scene[scene_id] = np.max(rel_pens_scene) if len(rel_pens_scene) > 0 else np.nan
        max_abs_pens_scene[scene_id] = np.max(abs_pens_scene) if len(abs_pens_scene) > 0 else np.nan


        avg_rel_pens_im[scene_id] = {}
        avg_abs_pens_im[scene_id] = {}
        max_rel_pens_im[scene_id] = {}
        max_abs_pens_im[scene_id] = {}
        for im_id, target in scene_targets.items():
            rel_pens_im = [e['errors']['rel_pen'] for e in scene_errs if e['im_id'] == im_id]
            abs_pens_im = [e['errors']['pen'] for e in scene_errs if e['im_id'] == im_id]
            avg_rel_pens_im[scene_id][im_id] = np.mean(rel_pens_im) if len(rel_pens_im) > 0 else np.nan
            avg_abs_pens_im[scene_id][im_id] = np.mean(abs_pens_im) if len(abs_pens_im) > 0 else np.nan
            max_rel_pens_im[scene_id][im_id] = np.max(rel_pens_im) if len(rel_pens_im) > 0 else np.nan
            max_abs_pens_im[scene_id][im_id] = np.max(abs_pens_im) if len(abs_pens_im) > 0 else np.nan

        for e in scene_errs:
            if e['obj_id'] not in rel_pens_obj.keys():
                rel_pens_obj[e['obj_id']] = []
            rel_pens_obj[e['obj_id']].append(e['errors']['rel_pen'])

            if e['obj_id'] not in abs_pens_obj.keys():
                abs_pens_obj[e['obj_id']] = []
            abs_pens_obj[e['obj_id']].append(e['errors']['pen'])

    avg_rel_pen = np.mean(list(avg_rel_pens_scene.values()))
    avg_abs_pen = np.mean(list(avg_abs_pens_scene.values()))
    max_rel_pen = np.max(list(max_rel_pens_scene.values()))
    max_abs_pen = np.max(list(max_abs_pens_scene.values()))

    avg_rel_pens_obj = {k: np.mean(v) for k, v in rel_pens_obj.items()}
    avg_abs_pens_obj = {k: np.mean(v) for k, v in abs_pens_obj.items()}
    max_rel_pens_obj = {k: np.max(v) for k, v in rel_pens_obj.items()}
    max_abs_pens_obj = {k: np.max(v) for k, v in abs_pens_obj.items()}

    obj_rel_pen_str = ', '.join(['{}: {:.3f}'.format(i, s) for i, s in avg_rel_pens_obj.items()])
    obj_abs_pen_str = ', '.join(['{}: {:.3f}'.format(i, s) for i, s in avg_abs_pens_obj.items()])

    scene_rel_pen_str = ', '.join(['{}: {:.3f}'.format(i, s) for i, s in avg_rel_pens_scene.items()])
    scene_abs_pen_str = ', '.join(['{}: {:.3f}'.format(i, s) for i, s in avg_abs_pens_scene.items()])

    scores = {
        'avg_rel_pen': avg_rel_pen,
        'avg_abs_pen': avg_abs_pen,
        'avg_obj_rel_pen': avg_rel_pens_obj,
        'avg_obj_abs_pen': avg_abs_pens_obj,
        'avg_scene_rel_pen': avg_rel_pens_scene,
        'avg_scene_abs_pen': avg_abs_pens_scene,
        'avg_im_rel_pen': avg_rel_pens_im,
        'avg_im_abs_pen': avg_abs_pens_im,
        'max_rel_pen': max_rel_pen,
        'max_abs_pen': max_abs_pen,
        'max_obj_rel_pen': max_rel_pens_obj,
        'max_obj_abs_pen': max_abs_pens_obj,
        'max_scene_rel_pen': max_rel_pens_scene,
        'max_scene_abs_pen': max_abs_pens_scene,
        'max_im_rel_pen': max_rel_pens_im,
        'max_im_abs_pen': max_abs_pens_im,
        'max_rel_pen_recall': {},
        'avg_rel_pen_recall': {},
    }

    max_im_rel_pens = np.array(sum([list(v.values()) for v in scores['max_im_rel_pen'].values()], []))
    avg_im_rel_pens = np.array(sum([list(v.values()) for v in scores['avg_im_rel_pen'].values()], []))

    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        scores['max_rel_pen_recall'][k] = (max_im_rel_pens < k).sum() / (
            ~np.isnan(max_im_rel_pens)).sum()
        scores['avg_rel_pen_recall'][k / 10] = (avg_im_rel_pens < k / 10).sum() / (
            ~np.isnan(avg_im_rel_pens)).sum()

    scores_path = p['out_scores_tpath'].format(eval_path=p['eval_path'], error_dir_path=error_dir_path,
                                               score_sign=score_sign)
    inout.save_json(scores_path, scores)

    misc.log('')
    misc.log('Average relative penetration: {:f}'.format(avg_rel_pen))
    misc.log('Average absolute penetration: {:f}'.format(avg_abs_pen))
    misc.log('Object average relative penetrations:\n {}'.format(obj_rel_pen_str))
    misc.log('Object average absolute penetrations:\n {}'.format(obj_abs_pen_str))
    misc.log('Scene average relative penetrations:\n {}'.format(scene_rel_pen_str))
    misc.log('Scene average absolute penetrations:\n {}'.format(scene_abs_pen_str))
    time_total = time.time() - time_start
    misc.log('Score calculation took {}s.'.format(time_total))
misc.log('Done.')
