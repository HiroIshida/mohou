from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "numpy",
    "psutil",
    "sklearn",
    "torch",
    "torchvision",
    "tinyfk>=0.4.8",
    "tqdm",
    "matplotlib",
    "albumentations",
    "opencv-python",
    "pybullet",
    'imageio==2.15.0;python_version<"3.7"',  # dependency of moviepy
    'cached_property;python_version<"3.8"',
    "moviepy",
    "PyYAML>=5.1",
    "types-PyYAML"
]

setup(
    name='mohou',
    version='0.1.0',
    description='Visuomotor imitation learning framework',
    author='Hirokazu Ishida',
    author_email='h-ishida@jsk.imi.i.u-tokyo.ac.jp',
    license='MIT',
    install_requires=install_requires,
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'mohou': ['py.typed']}
)
