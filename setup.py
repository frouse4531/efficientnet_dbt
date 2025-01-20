import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dbtnet',
    version=os.environ.get('PACKAGE_VERSION', '1.0.0'),
    description='DBTNet',
    author = 'Forest Rouse',
    packages=[],
    ext_modules=[],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    entry_points = {'console_scripts': ['softtriple=softtriple.lj_train:main']},
    include_package_data=True
)
