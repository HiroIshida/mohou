from setuptools import find_packages, setup

setup_requires = []

install_requires = [
    "natsort",
    "numpy",
    "psutil",
    "scikit-learn",
    "torch",
    "torchvision",
    "tinyfk<0.6",
    "tqdm",
    "matplotlib",
    "albumentations",
    'opencv-python-headless<4.3.0;python_version<"3.7"',  # because it takes too long to build
    'opencv-python-headless;python_version>="3.8"',
    "pybullet",
    'imageio==2.15.0;python_version<"3.7"',  # dependency of moviepy
    "moviepy",
    "PyYAML>=5.1",
    "types-PyYAML",
    "gdown",
]

setup(
    name="mohou",
    version="0.5.7",
    description="Visuomotor imitation learning framework",
    author="Hirokazu Ishida",
    author_email="h-ishida@jsk.imi.i.u-tokyo.ac.jp",
    url="https://github.com/HiroIshida/mohou",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
