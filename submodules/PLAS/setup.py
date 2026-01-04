from setuptools import setup, find_packages

setup(
    name='plas',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'kornia',
        'tqdm',

        # example/eval deps, maybe split?
        # sort_3d_gaussians
        'pandas',
        'plyfile',
        'trimesh',

        # sort_rgb_img
        'click',
        'pillow',


        # eval/compare_plas_flas
        'opencv-python',

        # eval/flas
        'lap',
        'matplotlib',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    package_data={
        'plas': ['../img/*.jpg'],
    },
    include_package_data=True,
)
