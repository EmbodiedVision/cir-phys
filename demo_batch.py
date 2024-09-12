#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ===================================================================
#
# This file is based on the demo.py file at
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/blob/c50df7816714007c7f2f5188995807b3b396ad3d/demo.py
# and was modified to process all object detections in an image as one batch.
#

import argparse
import json
from pathlib import Path

from cir.datasets.symmetries import make_se3

import colored_traceback
import gin
import imageio
import numpy as np
import torch
from gin.torch import external_configurables
from lietorch import SE3

from cir.crops import crop_inputs
from cir.detector import load_detector
from cir.pose_models import load_efficientnet
from test_batch import load_sdfs, visualize_meshes, generate_pose_from_detections, format_gin_override, load_raft_model
from cir.utils import Pytorch3DRenderer, get_perturbations
from cir.datasets import lin_interp

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def read_depth(depth_path: Path, depth_scale: float, interpolate=False):
    depth = np.array(imageio.imread(depth_path).astype(np.float32))
    depth = depth * depth_scale / 1000
    if interpolate:  # interpolating the missing depth values takes about 0.7s, scipy is slow
        return lin_interp(depth)
    return depth


@torch.no_grad()
def main():
    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=Path, required=True,
                        help="A folder with an rgb/ subdir, a scene_camera.json, and (Optionally) a depth/ subdir")
    parser.add_argument('--output_dir', type=Path, default="qualitative_output",
                        help="The directory to save qualitative output")
    parser.add_argument('-o', '--override', nargs='+', type=str, default=[], help="gin-config settings to override")
    parser.add_argument('--load_weights', type=str, required=True, help='path to the model weights to load')
    parser.add_argument('--num_outer_loops', type=int, default=2, help="number of outer-loops in each forward pass")
    parser.add_argument('--num_inner_loops', type=int, default=10, help="number of inner-loops in each forward pass")
    parser.add_argument('--num_solver_steps', type=int, default=3,
                        help="number of BD-PnP solver steps per inner-loop (doesn't affect Modified BD-PnP)")
    parser.add_argument('--obj_models', required=True,
                        choices=['ycbv', 'tless', 'lmo', 'hb', 'tudl', 'icbin', 'itodd', 'synphys', 'synpick'],
                        help="which object models to use")
    parser.add_argument('--intersection', action='store_true', help="Whether to use the penetration penalty")
    parser.add_argument('--warmup_iterations', type=int, default=1, help='How many outer iterations to use as warmup without intersection penalty')
    args = parser.parse_args()
    args.override = format_gin_override(args.override)
    gin.parse_config_files_and_bindings(
        ["configs/base.gin", f"configs/test_{args.obj_models}_rgbd.gin"], args.override)

    detector = load_detector()

    run_efficientnet = load_efficientnet()

    Pytorch3DRenderer()  # Loading Renders. This gets cached so it's only slow the first time.
    sdfs = load_sdfs(Pytorch3DRenderer().ds_dir / Pytorch3DRenderer().dataset_name / 'models') \
        if args.intersection else None

    model = load_raft_model(args.load_weights, intersection=args.intersection, sdfs=sdfs)
    model.eval()

    print(f"\n\nSaving images output to {args.output_dir}/\n\n")
    args.output_dir.mkdir(exist_ok=True)

    if not args.scene_dir.exists():
        raise FileNotFoundError(
            f"The directory {args.scene_dir} doesn't exist. Download a sample scene using ./download_sample.sh or set --scene_dir to a BOP scene directory.")
    if not (args.scene_dir / "rgb").exists():
        raise FileNotFoundError(f"The directory {args.scene_dir / 'rgb'} doesn't exist.")
    if not (args.scene_dir / "scene_camera.json").exists():
        raise FileNotFoundError(f"The file {args.scene_dir / 'scene_camera.json'} doesn't exist.")

    scene_cameras = json.loads((args.scene_dir / "scene_camera.json").read_text())
    image_loop = list(scene_cameras.items())
    # np.random.default_rng(0).shuffle(image_loop)
    scene_gt = json.loads((args.scene_dir / "scene_gt.json").read_text())

    for image_index, (frame_id, scene_camera) in enumerate(image_loop):
        camera_intrinsics = torch.as_tensor(scene_camera['cam_K'], device='cuda', dtype=torch.float32).view(1, 3, 3)
        TC0 = np.array(scene_camera['cam_t_w2c']) / 1000.
        RC0 = np.array(scene_camera['cam_R_w2c']).reshape(3, 3)
        TWC = make_se3(RC0, TC0)[None].to(device='cuda', dtype=torch.float32).inv()

        depth_scale = scene_camera['depth_scale']
        rgb_path = args.scene_dir / "rgb" / f"{int(frame_id):06d}.jpg"
        images = imageio.imread(rgb_path)
        if images.shape[2] > 3:
            images = images[:, :, :3]
        # render_resolution = torch.tensor(images.shape[:2], device='cuda', dtype=torch.float32).view(1, 2) / 2
        render_resolution = torch.tensor([240, 320], dtype=torch.float32).view(1, 2)
        images = torch.as_tensor(images, device='cuda', dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255

        depth_path = args.scene_dir / "depth" / f"{int(frame_id):06d}.png"
        interpolated_depth = read_depth(depth_path, depth_scale, interpolate=True)
        interpolated_depth = torch.as_tensor(interpolated_depth, device='cuda', dtype=torch.float32).unsqueeze(0)

        imageio.imwrite(args.output_dir / f"{image_index}.1_5_Image.png",
                        images[0].permute(1, 2, 0).mul(255).byte().cpu())

        # Generate candidate detections using a Mask-RCNN
        detections = detector.get_detections(images=images, detection_th=0.95)
        if len(detections) == 0:
            imageio.imwrite(args.output_dir / f"{image_index}.1_3_CIR_Outer-Loop-1.png",
                            images[0].new_zeros(images[0].shape).permute(1, 2, 0).mul(255).byte().cpu())
            continue

        # Convert the predicted bounding boxes to initial translation estimates
        data_TCO_init = generate_pose_from_detections(detections=detections, K=camera_intrinsics)

        # Crop the image given the translation predicted by the Mask-RCNN
        images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(
            images=images.expand(len(detections), -1, -1, -1), K=camera_intrinsics.expand(len(detections), -1, -1),
            TCO=data_TCO_init.poses, \
            labels=list(detections.infos.label), masks=detections.masks,
            sce_depth=interpolated_depth.expand(len(detections), -1, -1),
            render_size=render_resolution.squeeze().cpu().numpy())

        mrcnn_rendered_rgb, _, _ = Pytorch3DRenderer()(list(detections.infos.label), data_TCO_init.poses, K_cropped,
                                                       render_resolution)
        for obj_idx, render in enumerate(mrcnn_rendered_rgb):
            basename = f"{image_index}.{obj_idx + 1}"
            imageio.imwrite(args.output_dir / f"{basename}_1_Mask_RCNN_Initial_Translation.png",
                            render.permute(1, 2, 0).mul(255).byte().cpu())

        # Generate a coarse pose estimate using an efficientnet
        assert (mrcnn_rendered_rgb.shape == images_cropped.shape)
        images_input = torch.cat((images_cropped, mrcnn_rendered_rgb), dim=1)
        current_pose_est = run_efficientnet(images_input, data_TCO_init.poses, K_cropped)

        efficientnet_rendered_rgb, _, _ = Pytorch3DRenderer()(list(detections.infos.label), current_pose_est, K_cropped,
                                                              render_resolution)
        for obj_idx, render in enumerate(efficientnet_rendered_rgb):
            basename = f"{image_index}.{obj_idx + 1}"
            imageio.imwrite(args.output_dir / f"{basename}_2_Efficientnet_Prediction.png",
                            render.permute(1, 2, 0).mul(255).byte().cpu())

        labels = list(detections.infos.label)
        if Pytorch3DRenderer().dataset_name == 'synpick':
            labels += [None]
        visualize_meshes(labels, torch.cat([current_pose_est, TWC.inv().matrix()], dim=0),
                         Pytorch3DRenderer().mesh_lookup, camera_intrinsics[0],
                         path=args.output_dir / "render_efficientnet.png", alt_pose=True,
                         use_pyrender=True)


        # Visualize GT
        gt_labels = [f'obj_{o["obj_id"]:06d}' for o in scene_gt[frame_id]] + [None]
        gt_poses = torch.zeros_like(current_pose_est)
        gt_poses[:, -1, -1] = 1.
        for i, o in enumerate(scene_gt[frame_id]):
            gt_poses[i, :3, :3] = gt_poses.new_tensor(o['cam_R_m2c']).reshape(3, 3)
            gt_poses[i, :3, 3] = gt_poses.new_tensor(o['cam_t_m2c']) / 1000.
        visualize_meshes(gt_labels, torch.cat([gt_poses, TWC.inv().matrix()], dim=0),
                         Pytorch3DRenderer().mesh_lookup, camera_intrinsics[0],
                         path=args.output_dir / "render_gt.png",
                         alt_pose=True, use_pyrender=True)

        for outer_loop_idx in range(args.num_outer_loops):
            # Crop image given the previous pose estimate
            images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(
                images=images.expand(len(detections), -1, -1, -1), K=camera_intrinsics.expand(len(detections), -1, -1),
                TCO=current_pose_est, labels=list(detections.infos.label), masks=detections.masks,
                sce_depth=interpolated_depth.expand(len(detections), -1, -1),
                render_size=render_resolution.squeeze().cpu().numpy())

            # Render additional viewpoints
            input_pose_multiview = get_perturbations(current_pose_est)  # .flatten(0,1)
            Nr = input_pose_multiview.shape[1]

            input_pose_multiview = input_pose_multiview.flatten(0, 1)

            label_rep = np.repeat(list(detections.infos.label), Nr)
            K_rep = K_cropped.repeat_interleave(Nr, dim=0)
            res_rep = render_resolution.repeat_interleave(Nr * len(detections), dim=0)
            rendered_rgb, rendered_depth, _ = Pytorch3DRenderer()(label_rep, input_pose_multiview, K_rep, res_rep)

            # Forward pass
            combine = lambda a, b: torch.cat((a.unflatten(0, (len(detections), Nr)), b.unsqueeze(1)), dim=1)
            images_input = combine(rendered_rgb, images_cropped)
            depths_input = combine(rendered_depth, depths_cropped)
            masks_input = combine(rendered_depth > 1e-3, masks_cropped)
            pose_input = combine(input_pose_multiview, current_pose_est)
            K_input = combine(K_rep, K_cropped)

            outputs = model(Gs=pose_input, images=images_input, depths_fullres=depths_input,
                            masks_fullres=masks_input, intrinsics_mat=K_input, labels=list(detections.infos.label),
                            num_solver_steps=args.num_solver_steps, num_inner_loops=args.num_inner_loops,
                            intersection=outer_loop_idx >= args.warmup_iterations and args.intersection,
                            G_tote=TWC.inv())
            current_pose_est = SE3(outputs['Gs'][-1].contiguous()[:, -1]).matrix()

            labels = list(detections.infos.label)
            if Pytorch3DRenderer().dataset_name == 'synpick':
                labels += [None]
            visualize_meshes(labels, torch.cat([current_pose_est, TWC.inv().matrix()], dim=0),
                             Pytorch3DRenderer().mesh_lookup, camera_intrinsics[0],
                             path=args.output_dir / f"render_cir_outer_loop_{outer_loop_idx}.png", alt_pose=True,
                             use_pyrender=True)

            efficientnet_rendered_rgb, _, _ = Pytorch3DRenderer()(list(detections.infos.label), current_pose_est,
                                                                  K_cropped, render_resolution)
            # efficientnet_rendered_rgb, _, _ = Pytorch3DRenderer()([obj_label], current_pose_est, camera_intrinsics,
            #                                                       render_resolution * 2)
            for obj_idx, render in enumerate(efficientnet_rendered_rgb):
                basename = f"{image_index}.{obj_idx + 1}"
                imageio.imwrite(args.output_dir / f"{basename}_3_CIR_Outer-Loop-{outer_loop_idx}.png",
                                render.permute(1, 2, 0).mul(255).byte().cpu())
        visualize_meshes(labels, torch.cat([current_pose_est, TWC.inv().matrix()], dim=0),
                         Pytorch3DRenderer().mesh_lookup, camera_intrinsics[0],
                         path=args.output_dir / f"render_cir_final.png", alt_pose=True,
                         use_pyrender=True)

        for obj_idx, crop in enumerate(images_cropped):
            basename = f"{image_index}.{obj_idx + 1}"
            imageio.imwrite(args.output_dir / f"{basename}_4_Image_Crop.png",
                            crop.permute(1, 2, 0).mul(255).byte().cpu())

        break


if __name__ == '__main__':
    main()
