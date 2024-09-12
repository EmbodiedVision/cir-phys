#
# This file originates from
# https://github.com/princeton-vl/Coupled-Iterative-Refinement/tree/c50df7816714007c7f2f5188995807b3b396ad3d, licensed
# under the MIT license (see CIR-LICENSE in the root folder of this repository).
#
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='corr_sampler',
    ext_modules=[
        CUDAExtension('corr_sampler', 
            sources=[
                'extensions/sampler.cpp', 
                'extensions/sampler_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-arch=sm_50',
                    '-gencode=arch=compute_50,code=sm_50',
                    '-gencode=arch=compute_52,code=sm_52',
                    '-gencode=arch=compute_60,code=sm_60',
                    '-gencode=arch=compute_61,code=sm_61',
                    '-gencode=arch=compute_70,code=sm_70',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90'
                ]
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

