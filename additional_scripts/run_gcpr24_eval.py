#
# Copyright (c) 2024 Max Planck Society
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_DIR = Path('.').resolve()
LOCAL_DATA_DIR = PROJECT_DIR / Path("local_data")
ADDITIONAL_SCRIPTS_DIR = PROJECT_DIR / Path('additional_scripts')
TOOLKIT_DIR = Path(os.environ['BOP_TOOLKIT_PATH'])
assert TOOLKIT_DIR.exists()
EVAL_SCRIPT_PATH = TOOLKIT_DIR / 'scripts/eval_bop19_pose.py'
PENETRATION_ERROR_SCRIPT_PATH = ADDITIONAL_SCRIPTS_DIR / 'eval_penetration_errors.py'
PENETRATION_SCORE_SCRIPT_PATH = ADDITIONAL_SCRIPTS_DIR / 'eval_penetration_scores.py'

sys.path.append(TOOLKIT_DIR.as_posix())
from bop_toolkit_lib import inout  # noqa


def tar_files_to_csv(dir_path, out_csv_path):
    assert dir_path.exists()
    files = list(dir_path.glob("*.tar"))
    print(f"there are {len(files)} .tar files in {dir_path}")
    assert len(files) > 0, dir_path
    files_with_data = []
    preds = []
    for file in tqdm(files, desc="Combining .tar files"):
        df = torch.load(file)['predictions']['maskrcnn_detections/refiner']
        start, end = map(int, re.compile(".*_([0-9]+)_([0-9]+)_.*").fullmatch(file.stem).groups())
        print(f"Converting results from frame {start} to frame {end}")
        preds += convert_results(df)
        files_with_data.append((start, end, file))
        # files_with_data.append((start, end, df, file))
    files_with_data = sorted(files_with_data)
    old_end = 0
    while len(files_with_data) > 0: # Checking that no files are missing
        start, end, file = files_with_data.pop(0)
        assert start == old_end, f"File is missing: {old_end}"
        old_end = end
    print("No files are missing")

    inout.save_bop_results(out_csv_path, preds)
    return out_csv_path


def main():
    parser = argparse.ArgumentParser('Bop evaluation')
    parser.add_argument('--tar_dir', type=str, required=True)
    parser.add_argument('--result_id', type=str, required=True)
    parser.add_argument('--dataset', default='', type=str, required=True)
    parser.add_argument('--split', default='test', type=str, required=False)
    parser.add_argument('--csv_dir', required=True, type=str)
    parser.add_argument('--eval_dir', required=True, type=str)
    parser.add_argument('--eval_bg_pen', action='store_true')
    args = parser.parse_args()

    assert '_' not in args.result_id, "Result ID must not contain \"_\""

    tar_path = Path(args.tar_dir)

    csv_path = Path(args.csv_dir)
    csv_path.mkdir(exist_ok=True, parents=True)
    eval_path = Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True, parents=True)

    ds_name = args.dataset
    split_name = args.split
    csv_path = csv_path / f'{args.result_id}_{ds_name}-{split_name.replace("_", "-")}.csv'
    csv_path.parent.mkdir(exist_ok=True)

    tar_files_to_csv(tar_path, csv_path)
    run_evaluation(csv_path, eval_path, args.split, args.eval_bg_pen)


def run_evaluation(csv_path, eval_path, split, eval_bg_pen):
    run_bop_evaluation(csv_path, eval_path, split_name=split)
    run_penetration_evaluation(csv_path, eval_path, split_name=split, eval_bg_pen=eval_bg_pen)


def convert_results(predictions):  # , method):
    preds = []
    for n in tqdm(range(len(predictions))):
        TCO_n = predictions.poses[n]
        t = TCO_n[:3, -1] * 1e3  # m -> mm conversion
        R = TCO_n[:3, :3]
        row = predictions.infos.iloc[n]
        obj_id = int(row.label.split('_')[-1])
        score = row.score
        time = row.time
        pred = dict(scene_id=row.scene_id,
                    im_id=row.view_id,
                    obj_id=obj_id,
                    score=score,
                    t=t, R=R, time=time)
        preds.append(pred)
    return preds


def run_bop_evaluation(filename, eval_path, split_name=None):
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = TOOLKIT_DIR.as_posix()
    myenv['BOP_PATH'] = (LOCAL_DATA_DIR / 'bop_datasets').as_posix()

    script_path = EVAL_SCRIPT_PATH
    subprocess_args = ['python', script_path.as_posix(),
                       '--results_path', filename.parent.as_posix(),
                       '--result_filenames', filename.name,
                       '--eval_path', eval_path.as_posix()]
    if split_name != 'test' and 'synpick' in filename.name:
        subprocess_args.append("--targets_filename")
        subprocess_args.append(f'{split_name}_targets_no_gripper.json')
    subprocess.call(subprocess_args, env=myenv, cwd=TOOLKIT_DIR.as_posix())


def run_penetration_evaluation(filename, eval_path, split_name=None, eval_bg_pen=False):
    myenv = os.environ.copy()
    myenv['PYTHONPATH'] = TOOLKIT_DIR.as_posix()
    myenv['BOP_PATH'] = (LOCAL_DATA_DIR / 'bop_datasets').as_posix()

    error_script_path = PENETRATION_ERROR_SCRIPT_PATH
    subprocess_args = ['python', error_script_path.as_posix(),
                       '--results_path', filename.parent.as_posix(),
                       '--result_filenames', filename.name,
                       '--eval_path', eval_path.as_posix()]

    if split_name != 'test' and 'synpick' in filename.name:
        subprocess_args.append("--targets_filename")
        subprocess_args.append(f'{split_name}_targets_no_gripper.json')
    if eval_bg_pen:
        subprocess_args.append("--eval_bg_pen")
    subprocess.call(subprocess_args, env=myenv, cwd=TOOLKIT_DIR.as_posix())

    score_script_path = PENETRATION_SCORE_SCRIPT_PATH
    subprocess_args = ['python', score_script_path.as_posix(),
                       '--eval_path', eval_path.as_posix(),
                       '--error_dir_paths', filename.stem + '/error=penetration']
    if split_name != 'test' and 'synpick' in filename.name:
        subprocess_args.append("--targets_filename")
        subprocess_args.append(f'{split_name}_targets_no_gripper.json')
    subprocess.call(subprocess_args, env=myenv, cwd=TOOLKIT_DIR.as_posix())


if __name__ == '__main__':
    main()
