#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import mesh2sdf
import trimesh


def main():
    parser = ArgumentParser()
    parser.add_argument('--input', required=True, help="Input mesh")
    parser.add_argument('--output', required=True, help="Output sdf file")
    parser.add_argument('--resolution', type=int, default=64, help="Resolution of the SDF grid.")
    parser.add_argument('--visualize', action='store_true', help='Wheter to visualize SDF cuts')

    args = parser.parse_args()

    mesh = trimesh.load(args.input)
    if not isinstance(mesh, trimesh.Trimesh):
        assert isinstance(mesh, trimesh.Scene) and len(mesh.geometry) == 1, "Input contains more than one mesh!"
        mesh = list(mesh.geometry.values())[0]
    mesh_orig = mesh.copy()

    # Mesh needs to be in range -1. to 1. (with padding)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.
    max_dim = np.max(mesh.bounds[1] - mesh.bounds[0])

    scale = 1.8 / max_dim
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)

    sdf = mesh2sdf.compute(mesh.vertices, mesh.faces, size=args.resolution)

    output_dir = Path(args.output).parent
    os.makedirs(output_dir, exist_ok=True)
    np.savez(args.output, scale=1. / scale, offset=center, sdf=sdf)

    if args.visualize:
        record_folder = output_dir / 'tmp'
        os.makedirs(record_folder)

        for dim in range(len(sdf.shape)):
            sdf_transposed = np.transpose(sdf, [(dim + j) % len(sdf.shape) for j in range(len(sdf.shape))])
            for i, layer in enumerate(sdf_transposed):
                fig = plt.figure()
                fig.set_size_inches(5, 5)
                im = plt.imshow(layer, cmap=matplotlib.colormaps['seismic'], vmin=-1., vmax=1.)
                plt.colorbar(im)

                fig.tight_layout()
                fig.savefig(record_folder / f'{dim}_{i:03d}.png')
                plt.close(fig)

                fig = plt.figure()
                fig.set_size_inches(5, 5)
                im = plt.imshow(layer < 0.)
                plt.colorbar(im)

                fig.tight_layout()
                fig.savefig(record_folder / f'{dim}_thresh_{i:03d}.png')
                plt.close(fig)

        os.system('ffmpeg ' + ' '.join([f'-i {record_folder}/{dim}_%03d.png' for dim in range(len(sdf.shape))]) \
                  + ' ' + ' '.join([f'-i {record_folder}/{dim}_thresh_%03d.png' for dim in range(len(sdf.shape))])
                  + ' -filter_complex "' + ''.join([f'[{dim}]' for dim in range(len(sdf.shape))]) \
                  + f'hstack={len(sdf.shape)}[top];' \
                  + ''.join([f'[{dim + 3}]' for dim in range(len(sdf.shape))]) \
                  + f'hstack={len(sdf.shape)}[bottom];' \
                  + '[top][bottom]vstack[out]"' \
                  + ' -map [out] ' + args.output + '.mp4')
        shutil.rmtree(record_folder)


if __name__ == "__main__":
    main()
