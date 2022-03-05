from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "numpy",
    "psutil",
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
    version='0.0.3',
    description='Visuomotor imitation learning framework',
    author='Hirokazu Ishida',
    author_email='h-ishida@jsk.imi.i.u-tokyo.ac.jp',
    license='MIT',
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'mohou': ['py.typed']}
)
