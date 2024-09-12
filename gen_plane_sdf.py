#
# Copyright (c) 2024 Max-Planck-Gesellschaft
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

resolution = 64

sdfs_z = np.linspace(-1, 1, resolution)

sdfs = np.repeat(np.repeat(sdfs_z[None, None, :], resolution, axis=0), resolution, axis=1)

np.savez('plane.npz', sdf=sdfs, scale=0.5, offset=np.zeros(3))
