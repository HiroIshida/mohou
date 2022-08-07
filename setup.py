from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "natsort",
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
    "moviepy",
    "PyYAML>=5.1",
    "types-PyYAML",
]

setup(
    name="mohou",
    version="0.3.10",
    description="Visuomotor imitation learning framework",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=install_requires,
    packages=find_packages(exclude=("tests", "docs")),
    package_data={"mohou": ["py.typed"]},
)
