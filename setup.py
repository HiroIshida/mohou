import sys
from setuptools import setup, find_packages

setup_requires = []

install_requires = [
        "numpy",
        "torch",
        "torchvision",
        "tinyfk",
        "tqdm",
        "matplotlib",
        "albumentations"
        ]

# for running demo
extras_require = {
        'test': ["pybullet", "moviepy"]
        }

setup(
    name='mohou',
    version='0.0.0',
    description='',
    license=license,
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(exclude=('tests', 'docs'))
)
