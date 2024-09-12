#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# ===================================================================
#
# This file is based on the test.py file at
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/blob/c50df7816714007c7f2f5188995807b3b396ad3d/test.py
# and was modified to process all object detections in an image as one batch.
#
import argparse
import ast
import io
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import colored_traceback
import gin
import imageio
import numpy as np
import pandas as pd
import pyrender
import pytorch3d.transforms
import torch
from PIL import Image
from gin.torch import external_configurables
from lietorch import SE3
from pyrender import RenderFlags
from tqdm import tqdm

from cir.crops import crop_inputs
from cir.datasets import BOPDataset, collate_fn
from cir.detector import PandasTensorCollection, concatenate, load_detector
from cir.pose_models import load_efficientnet, RaftSe3
from cir.utils import Pytorch3DRenderer, get_perturbations, transform_pts, mat2SE3

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def format_gin_override(overrides):
    if len(overrides) > 0:
        print("Overriden parameters:", overrides)
    output = deepcopy(overrides)
    for i, o in enumerate(overrides):
        k, v = o.split('=')
        try:
            ast.literal_eval(v)
        except:
            output[i] = f'{k}="{v}"'
    return output


@gin.configurable
def gin_globals(**kwargs):
    return SimpleNamespace(**kwargs)


def make_datasets(dataset_splits):
    datasets = []
    for (kwargs, n_repeat) in dataset_splits:
        ds = BOPDataset(**kwargs)
        print(f'Loaded BOPDataset({kwargs}) with {len(ds)}x{n_repeat}={len(ds) * n_repeat} images.')
        datasets.extend([ds] * n_repeat)
    return torch.utils.data.ConcatDataset(datasets)


def create_dataloader(dataset, batch_size, world_size, rank, num_workers, training):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=training, num_replicas=world_size, rank=rank, drop_last=training)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, persistent_workers=(num_workers > 0),
                                       num_workers=num_workers, sampler=sampler, pin_memory=True, collate_fn=collate_fn,
                                       drop_last=training)


def load_sdfs(sdf_path, no_bg=False):
    sdfs = {}
    if 'synpick' in sdf_path.as_posix() and not no_bg:
        bg_sdf = dict(np.load(sdf_path / 'tote.npz'))
        sdfs['bg'] = {k: torch.tensor(v).float().cuda() for k, v in bg_sdf.items()}
    elif 'ycbv' in sdf_path.as_posix() and not no_bg:
        bg_sdf = dict(np.load(sdf_path / 'plane.npz'))
        sdfs['bg'] = {k: torch.tensor(v).float().cuda() for k, v in bg_sdf.items()}
    for file in sdf_path.glob("obj_*.npz"):
        id = file.stem
        sdf = dict(np.load(file))
        sdfs[id] = {k: torch.tensor(v).float().cuda() for k, v in sdf.items()}
        sdfs[id]['scale'] /= 1000.
        sdfs[id]['offset'] /= 1000.
    return sdfs


def load_raft_model(file_path, intersection=False, sdfs=None, sample_pcis=False, generator=None, low_res_grid_factor=1):
    model = RaftSe3(intersection=intersection, sdfs=sdfs, sample_pcis=sample_pcis, generator=generator,
                    low_res_grid_factor=low_res_grid_factor)
    if file_path:
        state_dict = torch.load(file_path)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"Loaded RAFT model: {file_path}")
    model.cuda()
    return model


@gin.configurable
def gt_masks_to_detections(masks_gt, obs, min_visib=-1.0):
    infos = []
    masks = []
    bboxes = []
    for obj in obs['objects'][0]:
        if obj['visib_fract'] <= min_visib:
            continue
        bbox = torch.tensor(obj['bbox'])
        info = dict(
            batch_im_id=0,
            label=obj['label'],
            score=obj['visib_fract'],
        )
        mask = masks_gt[0] == obj['id_in_segm']
        bboxes.append(bbox)
        masks.append(mask)
        infos.append(info)

    if len(bboxes) > 0:
        bboxes = torch.stack(bboxes).cuda().float()
        masks = torch.stack(masks).cuda()
    else:
        infos = dict(score=[], label=[], batch_im_id=[])
        bboxes = torch.empty(0, 4).cuda().float()
        masks = torch.empty(0, masks_gt.shape[0], masks_gt.shape[1], dtype=torch.bool).cuda()

    detections = PandasTensorCollection(
        infos=pd.DataFrame(infos),
        bboxes=bboxes,
        masks=masks,
    )

    sorted_order = detections.infos.sort_values('score', ascending=False).index
    detections = detections[sorted_order]

    return detections


def align_pointclouds_to_boxes(boxes_2d, model_points_3d, K):
    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    bsz = boxes_2d.shape[0]
    z_guess = 1.0
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    TCO = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [-1, 0, 0, z_guess],
        [0, 0, 0, 1]
    ]).to(torch.float).to(boxes_2d.device).repeat(bsz, 1, 1)
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    xy_init = ((bb_xy_centers - cxcy) * z_guess) / fxfy
    TCO[:, :2, 3] = xy_init

    C_pts_3d = transform_pts(TCO, model_points_3d)
    deltax_3d = C_pts_3d[:, :, 0].max(dim=1).values - C_pts_3d[:, :, 0].min(dim=1).values
    deltay_3d = C_pts_3d[:, :, 1].max(dim=1).values - C_pts_3d[:, :, 1].min(dim=1).values

    bb_deltax = (boxes_2d[:, 2] - boxes_2d[:, 0]) + 1
    bb_deltay = (boxes_2d[:, 3] - boxes_2d[:, 1]) + 1

    z_from_dx = fxfy[:, 0] * deltax_3d / bb_deltax
    z_from_dy = fxfy[:, 1] * deltay_3d / bb_deltay
    z = (z_from_dy.unsqueeze(1) + z_from_dx.unsqueeze(1)) / 2

    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    TCO[:, :2, 3] = xy_init
    TCO[:, 2, 3] = z.flatten()
    return TCO


@gin.configurable
def generate_pose_from_detections(renderer, detections, K):
    K = K[detections.infos['batch_im_id'].values]
    boxes = detections.bboxes
    points_3d = renderer.get_pointclouds(detections.infos['label'])
    TCO_init = align_pointclouds_to_boxes(boxes, points_3d, K)
    return PandasTensorCollection(infos=detections.infos, poses=TCO_init)


def format_results(predictions):
    df = defaultdict(list)
    df = pd.DataFrame(df)
    results = dict(summary=dict(), summary_txt='',
                   predictions=predictions, metrics=dict(),
                   summary_df=df, dfs=dict())
    return results


def animate_rotation(pr_scene, renderer, cam_trans, path, steps=240):
    R = torch.eye(4)
    R[:3, :3] = pytorch3d.transforms.axis_angle_to_matrix(torch.tensor([0, 0, 2 * math.pi / steps]))
    for step in tqdm(range(steps), desc=f'Rendering to {path.name}'):
        frame_path = path.parent / path.name.format(step)
        pr_scene.set_pose(pr_scene.main_camera_node, pose=cam_trans.numpy())
        color, _ = renderer.render(pr_scene, RenderFlags.SHADOWS_ALL)
        image = Image.fromarray(color)
        image.save(frame_path)
        cam_trans = R @ cam_trans


def visualize_meshes(labels, poses, mesh_lookup, camera_intrinsics, resolution=(1920, 1080), path=None, alt_pose=False,
                     use_pyrender=True, animate=False, z_near=0.01):
    import trimesh
    import matplotlib

    cm = matplotlib.cm.tab10(range(len(poses)))
    cm[:, 3] = 0.4

    cam = trimesh.scene.Camera(resolution=resolution,
                               focal=(camera_intrinsics[0, 0].cpu(), camera_intrinsics[1, 1].cpu()), z_near=z_near)
    if labels[-1] is None:
        cam_trans = poses[-1].inverse().cpu() @ torch.tensor([[1, 0, 0, 0],
                                                              [0, -1, 0, 0],
                                                              [0, 0, -1, 0],
                                                              [0, 0, 0, 1.]])
    else:
        cam_trans = np.eye(4)

    scene = trimesh.Scene(camera=cam, camera_transform=cam_trans)
    for label, pose, c in zip(labels, poses, cm):
        if label:
            id = int(label[-6:]) - 1
            image = Image.fromarray((mesh_lookup.textures.maps_list()[id].cpu().numpy() * 255).astype(np.uint8))
            uvs = mesh_lookup.textures.verts_uvs_list()[id].cpu()
            mat = trimesh.visual.texture.SimpleMaterial(image=image)
            color_visuals = trimesh.visual.TextureVisuals(uv=uvs, image=image, material=mat)
            mesh = trimesh.Trimesh(mesh_lookup.verts_list()[id].cpu(), mesh_lookup.faces_list()[id].cpu(),
                                   visual=color_visuals)
        else:
            mesh = list(trimesh.load(
                Pytorch3DRenderer().ds_dir / Pytorch3DRenderer().dataset_name / 'models/tote.glb').geometry.values())[0]
            color_vis = mesh.visual.to_color()
            color_vis.vertex_colors[3] = 100
            color_vis.vertex_colors = color_vis.vertex_colors.repeat(len(mesh.vertices)).reshape(4,
                                                                                                 len(mesh.vertices)).T
            mesh.visual = color_vis
        scene.add_geometry(mesh, transform=(poses[-1].inverse() @ pose).cpu())

    if use_pyrender:
        pr_scene = pyrender.Scene.from_trimesh_scene(scene, ambient_light=[0.1, 0.1, 0.1])
        pr_cam = pyrender.IntrinsicsCamera(camera_intrinsics[0, 0].cpu(), camera_intrinsics[1, 1].cpu(),
                                           camera_intrinsics[0, 2].cpu(), camera_intrinsics[1, 2].cpu(),
                                           znear=cam.z_near)
        pr_light = pyrender.DirectionalLight([1., 1., 1.], intensity=5.0)
        pr_scene.add(pr_cam, pose=cam_trans)
        pr_scene.add(pr_light)
        for mesh in list(pr_scene.meshes):
            if mesh.primitives[0].material.baseColorFactor[0] != mesh.primitives[0].material.baseColorFactor[1]:
                continue
            mesh.primitives[0].material.baseColorFactor = np.array([0.8, 0.8, 0.8, 1.])
            mesh.primitives[0].material.metallicFactor = 0.0
            mesh.primitives[0].material.roughnessFactor = 0.5

    if path:
        if use_pyrender:
            renderer = pyrender.OffscreenRenderer(resolution[0], resolution[1])
            if animate:
                animate_rotation(pr_scene, renderer, cam_trans, path.parent / (path.stem + '_{:03d}' + path.suffix))
            else:
                color, _ = renderer.render(pr_scene, RenderFlags.SHADOWS_ALL)
                image = Image.fromarray(color)
        else:
            bytes = scene.save_image()
            image = Image.open(io.BytesIO(bytes))
        image.save(path)
        if alt_pose:
            # cam_trans = torch.tensor([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]]) @ cam_trans
            cam_trans = torch.tensor(
                [[1., 0., 0., -0.00147708],
                 [0., 0., -1., -1.17939451],
                 [0., 1., 0., 0.07862591],
                 [0., 0., 0., 1.]]
            )
            scene.camera_transform = cam_trans
            if use_pyrender:
                if animate:
                    animate_rotation(pr_scene, renderer, cam_trans, path.parent / (path.stem + '_alt_{:03d}' + path.suffix))
                else:
                    pr_scene.set_pose(pr_scene.main_camera_node, pose=cam_trans.numpy())
                    color, _ = renderer.render(pr_scene, RenderFlags.SHADOWS_ALL)
                    image = Image.fromarray(color)
            else:
                bytes = scene.save_image()
                image = Image.open(io.BytesIO(bytes))
            alt_path = path.parent / (path.stem + "_alt" + path.suffix)
            image.save(alt_path)
    else:
        if use_pyrender:
            pyrender.Viewer(pr_scene, viewport_size=resolution)
        else:
            scene.show()
            print(scene.camera_transform)
            print(scene.camera.z_near)


@torch.no_grad()
def main():
    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--override', nargs='+', type=str, default=[], help="gin-config settings to override")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--load_weights', type=str, default=None, help='path to the model weights to load')
    parser.add_argument('--num_outer_loops', type=int, default=4, help="number of outer-loops in each forward pass")
    parser.add_argument('--num_inner_loops', type=int, default=40, help="number of inner-loops in each forward pass")
    parser.add_argument('--num_solver_steps', type=int, default=10,
                        help="number of BD-PnP solver steps per inner-loop (doesn't affect Modified BD-PnP)")
    parser.add_argument('--save_dir', type=Path, default="test_evaluation")
    parser.add_argument('--dataset', required=True,
                        choices=['ycbv', 'tless', 'lmo', 'hb', 'tudl', 'icbin', 'itodd', 'synpick-pick-targeted',
                                 'synpick-pick-untargeted'],
                        help="dataset for training (and evaluation)")
    parser.add_argument('--intersection', action='store_true', help="Whether to use the penetration penalty")
    parser.add_argument('--warmup_iterations', type=int, default=1,
                        help='How many outer iterations to use as warmup without intersection penalty')
    parser.add_argument('--sample_pcis', action='store_true',
                        help='Whether to sample pcis for intersection (instead of using the entire grid)')
    parser.add_argument('--use_gt_masks', action='store_true',
                        help="use the GT masks instead of Mask R-CNN")
    parser.add_argument('--max_detections', type=int, default=-1, help='Limit how many detections will be processed (default=-1 = all detections)')
    parser.add_argument('--no_bg', action='store_true', help='Do not use the background as additional cue on synpick or ycbv')
    parser.add_argument('--low_res_grid_factor', type=int, default=1, help="Whether to downsample the grid (by a factor of 2, 4, 8)")
    parser.add_argument('--debug_vis', action='store_true', help='Whether to display debug visualizations.')

    args = parser.parse_args()
    args.override = format_gin_override(args.override)
    gin.parse_config_files_and_bindings(
        ["configs/base.gin", f"configs/test_{args.dataset}_rgbd.gin"], args.override)
    test_dataset = make_datasets(gin_globals().test_splits)
    print(f"The entire dataset is of length {len(test_dataset)}")

    if 'SLURM_ARRAY_TASK_MIN' in os.environ:  # array job
        assert int(os.environ['SLURM_ARRAY_TASK_MIN']) == 0
        num_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        assert len(test_dataset) % num_jobs == 0
        num_images = len(test_dataset) // num_jobs
        start_index = int(os.environ['SLURM_ARRAY_TASK_ID']) * num_images
    else:
        num_images = args.num_images if (args.num_images is not None) else len(test_dataset)
        start_index = args.start_index
    print(f"Processing images in range [{start_index}, {start_index + num_images})")
    if not args.use_gt_masks:
        detector = load_detector()
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(start_index, start_index + num_images)))
    assert len(test_dataset) == num_images, len(test_dataset)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    qual_output = args.save_dir / "qual_output"
    qual_output.mkdir(parents=True, exist_ok=True)

    test_loader = create_dataloader(test_dataset, 1, 1, 0, num_workers=0, training=False)

    run_efficientnet = load_efficientnet()

    Pytorch3DRenderer()  # Loading Renders. This gets cached so it's only slow the first time.
    sdfs = load_sdfs(Pytorch3DRenderer().ds_dir / Pytorch3DRenderer().dataset_name / 'models') \
        if args.intersection else None
    if "ycbv" in args.dataset and not args.no_bg:
        with open(Pytorch3DRenderer().ds_dir / Pytorch3DRenderer().dataset_name / 'sporeagent' /
                  'test_posecnn_plane.pkl', 'rb') as f:
            samples = pickle.load(f)

    model = load_raft_model(args.load_weights, intersection=args.intersection, sdfs=sdfs, sample_pcis=args.sample_pcis,
                            low_res_grid_factor=args.low_res_grid_factor)
    model.eval()

    all_preds = []
    for image_index, (images, masks_gt, obs) in enumerate(test_loader):
        start_time = time.time()
        print(f"Processing image {image_index + 1}/{num_images}")
        should_save_img = (random.random() < gin_globals().save_img_prob)
        images = images.to('cuda', torch.float).permute(0, 3, 1, 2) / 255
        obs['camera'] = {k: (v.to('cuda') if torch.is_tensor(v) else v) for k, v in obs['camera'].items()}
        TWC = mat2SE3(obs['camera']['TWC'])

        if args.use_gt_masks:
            detections = gt_masks_to_detections(masks_gt, obs)
        else:
            detections = detector.get_detections(images=images)

        if len(detections) == 0:
            continue

        if args.max_detections > 0:
            if len(detections) > args.max_detections:
                print(f'WARNING: detected {len(detections)} objects, but only using {args.max_detections}!',
                      file=sys.stderr)
            detections = detections[:args.max_detections]

        print(f"Processing {len(detections)} objects || {time.strftime('%l:%M:%S %p on %b %d, %Y')}\n")

        data_TCO_init = generate_pose_from_detections(detections=detections, K=obs['camera']['K'])
        scene_id, view_id = obs['frame_info']['scene_id'][0], obs['frame_info']['view_id'][0]
        data_TCO_init.infos.loc[:, "scene_id"] = scene_id
        data_TCO_init.infos.loc[:, "view_id"] = view_id
        data_TCO_init.infos.loc[:, "time"] = -1.0

        G_bg = None
        if "ycbv" in args.dataset and not args.no_bg:
            matching_samples = [s for s in samples if s['scene'] == scene_id and s['frame'] == view_id]
            assert len(matching_samples) == 1
            plane_pose = torch.tensor(matching_samples[0]['plane']).to(TWC.device)
            plane_pose[:3, 3] /= 1000. # Plane pose is in mm
            G_bg = mat2SE3(plane_pose.unsqueeze(0))
        elif "synpick" in args.dataset and not args.no_bg:
            G_bg = TWC.inv()

        images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(
            images=images.expand(len(detections), -1, -1, -1), K=obs['camera']['K'].expand(len(detections), -1, -1),
            TCO=data_TCO_init.poses, labels=list(detections.infos.label), masks=detections.masks,
            sce_depth=obs['camera']['interpolated_depth'].expand(len(detections), -1, -1), render_size=(240, 320))

        mrcnn_rendered_rgb, _, _ = Pytorch3DRenderer()(list(detections.infos.label), data_TCO_init.poses, K_cropped,
                                                       obs['camera']['resolution'].div(2))
        assert (mrcnn_rendered_rgb.shape == images_cropped.shape)
        images_input = torch.cat((images_cropped, mrcnn_rendered_rgb), dim=1)
        current_pose_est = run_efficientnet(images_input, data_TCO_init.poses, K_cropped)

        labels = list(detections.infos.label)
        if args.debug_vis:
            if Pytorch3DRenderer().dataset_name == 'synpick':
                labels += [None]
            visualize_meshes(labels,
                             torch.cat([current_pose_est, TWC.inv().matrix()], dim=0),
                             Pytorch3DRenderer().mesh_lookup, obs['camera']['K'][0])

        for outer_loop_idx in range(args.num_outer_loops):
            images_cropped, K_cropped, _, _, masks_cropped, depths_cropped = crop_inputs(
                images=images.expand(len(detections), -1, -1, -1), K=obs['camera']['K'].expand(len(detections), -1, -1),
                TCO=current_pose_est, labels=list(detections.infos.label), masks=detections.masks,
                sce_depth=obs['camera']['interpolated_depth'].expand(len(detections), -1, -1), render_size=(240, 320)
            )

            # Render additional viewpoints
            input_pose_multiview = get_perturbations(current_pose_est)  # .flatten(0,1)
            Nr = input_pose_multiview.shape[1]

            input_pose_multiview = input_pose_multiview.flatten(0, 1)

            label_rep = np.repeat(list(detections.infos.label), Nr)
            K_rep = K_cropped.repeat_interleave(Nr, dim=0)
            res_rep = obs['camera']['resolution'].div(2).repeat_interleave(Nr * len(detections), dim=0)
            rendered_rgb, rendered_depth, _ = Pytorch3DRenderer()(label_rep, input_pose_multiview, K_rep, res_rep)
            if should_save_img:
                for obj_idx, (_, obj_label, _) in detections.infos.iterrows():
                    basename = f"{scene_id}_{view_id}_{obj_label}_{obj_idx + 1}"
                    imageio.imwrite(qual_output / f"{basename}_B{outer_loop_idx}.png",
                                    rendered_rgb[obj_idx * Nr].permute(1, 2, 0).mul(255).byte().cpu())

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
                            G_bg=G_bg)
            current_pose_est = SE3(outputs['Gs'][-1].contiguous()[:, -1]).matrix()

            if args.debug_vis:
                labels = list(detections.infos.label)
                if Pytorch3DRenderer().dataset_name == 'synpick':
                    labels += [None]
                visualize_meshes(labels,
                                 torch.cat([current_pose_est, TWC.inv().matrix()], dim=0),
                                 Pytorch3DRenderer().mesh_lookup, obs['camera']['K'][0])

        data_TCO_init.infos.loc[:, "time"] = time.time() - start_time
        batch_preds = PandasTensorCollection(data_TCO_init.infos,
                                             poses=current_pose_est.cpu(),
                                             mrcnn_mask=detections.masks.cpu())
        all_preds.append(batch_preds)
        image_depth_cropped = PandasTensorCollection(infos=pd.DataFrame({'scene_id': [scene_id], 'view_id': [view_id]}),
                                                     image_cropped=images.cpu(),
                                                     interpolated_depth_cropped=obs['camera']['interpolated_depth'].cpu(),
                                                     depth_cropped=obs['camera']['orig_depth'].cpu(),
                                                     K_cropped=obs['camera']['K'].cpu())
        images_depths_cropped.append(image_depth_cropped)

        # Saving qualitative output
        if should_save_img:
            final_rendered_rgb, _, _ = Pytorch3DRenderer()(list(detections.infos.label), current_pose_est, K_cropped,
                                                           obs['camera']['resolution'].div(2))
            for obj_idx, (_, obj_label, _) in detections.infos.iterrows():
                basename = f"{scene_id}_{view_id}_{obj_label}_{obj_idx + 1}"
                imageio.imwrite(qual_output / f"{basename}_A.png",
                                mrcnn_rendered_rgb[obj_idx].permute(1, 2, 0).mul(255).byte().cpu())
                imageio.imwrite(qual_output / f"{basename}_C.png",
                                final_rendered_rgb[obj_idx].permute(1, 2, 0).mul(255).byte().cpu())
                imageio.imwrite(qual_output / f"{basename}_D.png",
                                images_cropped[obj_idx].permute(1, 2, 0).mul(255).byte().cpu())

    all_preds = {f'maskrcnn_detections/refiner': concatenate(all_preds),
                 f'inputs': concatenate(images_depths_cropped)}
    results = format_results(all_preds)
    output_filepath = args.save_dir / f'{gin_globals().dataset_name}_{start_index}_{start_index + num_images}_results.pth.tar'
    torch.save(results, output_filepath)
    print("Done.")


if __name__ == '__main__':
    main()
