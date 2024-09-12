#
# Copyright (c) 2024 Max Planck Society
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", required=True, help="The split to generate targets for")

    args = parser.parse_args()

    split_dir = Path(args.split)

    targets = []
    
    for scene_dir in sorted(split_dir.iterdir()):
        with open(scene_dir / "scene_gt.json") as f:
            scene_gt = json.load(f)
        n_frames = len(scene_gt)
        min_depths = {}
        for depth_frame in sorted((scene_dir / 'depth').glob('*.png')):
            depth = np.array(Image.open(depth_frame))
            min_depths[str(int(depth_frame.stem))] = depth.min()
        box_depth = max(min_depths.values())
        for frame_id, frame_gt in sorted(scene_gt.items(), key=lambda i: int(i[0])):
            if min_depths[frame_id] < box_depth and not int(frame_id) == 0:
                continue
            print(f'Adding targets for {scene_dir.name}, frame {frame_id}')
            obj_id_counts = {}
            for obj_gt in frame_gt:
                if obj_gt["obj_id"] not in obj_id_counts.keys():
                    obj_id_counts[obj_gt["obj_id"]] = 0
                obj_id_counts[obj_gt["obj_id"]] += 1
            for obj_id, count in obj_id_counts.items():
                targets.append({"im_id": int(frame_id), "inst_count": count, "obj_id": obj_id, "scene_id": int(scene_dir.name)})

    with open(args.split + "_targets_no_gripper.json", "w") as f:
        json.dump(targets, f)


if __name__ == "__main__":
    main()

