## mohou [![CI](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml) [![PypI Auto Release](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml) [![pypi-version](https://badge.fury.io/py/mohou.svg)](https://pypi.org/project/mohou/) [![python-version](https://img.shields.io/pypi/pyversions/mohou.svg)](https://pypi.org/project/mohou/)

This package implements conv-autoencoder based visuo-motor imitation learning using pytorch. This package focuses on extensibility. One can define custom data types (vector, image and composite-image) and also map/inverse from these custom types to feature vector that fed into the LSTM. Alongside the imitation learning framework, this package provides two demo using pybullet and rlbench to show example usage. Please try [kinematic pybullet demo](/pipeline/pybullet_demo.sh) and [dynamic rlbench demo](/pipeline/rlbench_demo.sh). The results of demo is available on [google drive](https://drive.google.com/drive/folders/1RQU76D5YpKuQ81AZfPMU1YlgIdNrliyt?usp=sharing).

The ros wrapper of this package can be found in [mohou_ros](https://github.com/HiroIshida/mohou_ros). Although `mohou_ros` currently supports only PR2 robot, many useful utilities and scripts for working with a real robot are included.

<img src="https://user-images.githubusercontent.com/38597814/160765355-b1d490a9-fb9b-40da-b1fb-d9eae476fda0.gif" width="50%" />

one of result of applying this framework to rlbench's task

## Instllation

For stable version:
```bash
pip3 install mohou
```

For beta from source:
```bash
git clone https://github.com/HiroIshida/mohou.git
cd mohou
pip install -e . 
```

## Introduction

### The example pipeline
After the pip install you are ready to start [pybullet demo](/pipeline/pybullet_demo.sh)! We also provide [rlbench_demo](/pipeline/rlbench_demo.sh). As for rlbench demo, additional installation step of pyrep and rlbench is required.  See https://github.com/stepjam/RLBench for the detail.

### concept of "project"
First, the important concept of the mohou package is "project". Each "project" has each directory and the directory contains everything, e.g. dataset, trained models, visualization results.  Thanks to this concept, hard-coding the file path of `TrainCache` and  `EpisodeBundle`, and many other stuff can be avoided. The use of the concept of project enables easy loading many objects. For example, `EpisodeBundle` which is a bundle of episodic sequential data is 
can be dumped and loaded by
```
EpisodeBundle.dump(project_path)
EpisodeBundle.load(project_path)
```

### pipeline
Except the visualization stuff, the pipeline consists of 1) generation of dataset, 2) training autoencoder, 3) trainling lstm, 4) execution using the trained policy. For example, in the `pybullet_demo.sh`, `kuka_reaching.py`, `python3 -m mohou.script.train_autoencoder`, `python3 -m mohou.script.train_lstm`, and `python3 $example_path/kuka_reaching.py --feedback` corresponds to the four core steps. The result of all trained model is saved in `{project_path}/models` directory.

Note that step 1 and step 4 must vary according to the problem and `kuka_reaching.py` is just an example. That is, if you use the real robot, you must write own dataset collection program and execution program. 

Other than the above steps, the software provide visualization method for autoencoder and lstm training reuslts, which are saved in `{project_path}/autoencoder_result` and `{project_path}/lstm_result` respectively.

The visualization of autoencoder result is done by
```bash
python3 -m mohou.script.visualize_autoencoder_result  # plus options
```
which plots comparison of original and reconstruction images side-by-side.

The visualization of lstm result is done by
```bash
python3 -m mohou.script.visualize_lstm_result # plus options
```
which plots the result of LSTM prediction of images as gif files and joint angles as png files. In the prediction, we first feeds some sequential state to LSTM and then propagate without feeding any extra images.

These visualization is extremely important to know the training quality. Based on the visualization result, you can decide increase the number of episode data or increase the training epoch.

### Data collection
The teaching data must be saved as a `EpisodeBundle`. `EpisodeBundle` consists of multiple `EpisodeData`. `EpisodeData` consists of multiple `ElementSequence`. And, `ElementSequence` consists of sequence of each elements like `AngleVector` and `RGBImage`. The pseudo-code of data collection looks like the below. 
```python
import numpy as np
from mohou.types import AngleVector, ElementSequence, EpisodeBundle, EpisodeData, RGBImage


def obtain_rgb_from_camera() -> np.ndarray:  # type: ignore
    # implement by your self
    pass


def obtain_joint_configuration() -> np.ndarray:  # type: ignore
    # implement by your self
    pass


def create_episode_data() -> EpisodeData:
    n_step = 100
    rgb_list = []
    av_list = []
    for _ in range(n_step):
        rgb_numpy = obtain_rgb_from_camera()
        av_numpy = obtain_joint_configuration()

        rgb = RGBImage(rgb_numpy, dtype=np.uint8)  # type: ignore
        av = AngleVector(av_numpy)

        rgb_list.append(rgb)
        av_list.append(av)

    # convering list of element to a sequence type
    rgb_sequence = ElementSequence(rgb_list)
    av_sequence = ElementSequence(av_list)
    return EpisodeData.from_seq_list([rgb_sequence, av_sequence])


n_episode = 20
episode_list = []
for _ in range(n_episode):
    episode_list.append(create_episode_data())
chunk = EpisodeBundle.from_data_list(episode_list)
chunk.dump(project_path)
```
Of course you can make a `EpisodeBundle` consists of `DepthImage` and `GripperState` or other your custom type.

### Execution
A pseudo-code for execution can be written as below:
```python
from mohou.default import create_default_propagator
from mohou.propagator import Propagator
from mohou.types import AngleVector, ElementDict, RGBImage


# create_default_propagator functions automatically resolve the autoencoder and lstm model path
# given the project_path, and then create the propagator.
propagator: Propagator = create_default_propagator(your_project_path)

while True:
    # Observation using real/simulated robot
    rgb: RGBImage = obtain_rgb_from_camera()  # define the function by yourself
    av: AngleVector = obtain_joint_configuration()  # define the function by yourself
    elem_dict = ElementDict((rgb, av))

    propagator.feed(elem_dict)

    # If your fed elem_dict contains RGBImage and AngleVector, then propagated
    # elem_dict_pred also has RGBImage and AngleVector
    elem_dict_pred: ElementDict = propagator.predict(n_prop=1)[0]

    # Get specific element by providing the elemen type as a key
    rgb_pred = elem_dict_pred[AngleVector]
    av_pred = elem_dict_pred[RGBImage]

    # send command
    send_next_angle_vector(av_pred)  # define by yourself
```

### Element type hierarchy and user's custom element type
The following figure show the type hierarchy. In the `mohou` framework, only the leaf types (filled by grey) can be instantiated. In most case, users would create custom type by inheriting from either `CompositeImageBase`, or `VectorBase` or `PrimitiveImageBase`. For the detail, please refere to [`mohou/types.py`](/mohou/types.py) for how the built-in concrete types such as `RGBDImage`, `RGBImage` and `AngleVector` are defined.
![graph](https://user-images.githubusercontent.com/38597814/183268967-65d133fd-3926-4e56-ac39-b398f30eb7bb.png)


## Contribution
When you make a new PR, you need to check that the tests passed and formatting is correct.

### testing
The test for `mohou` software consists of  3 steps: 1) static type check by mypy, 2) unit test by pytest, 3) integration test and 4) regression test.
To running these tests, you need install `mypy` and `pytest` by
```
pip3 install pytest mypy  # or use --user option 
```
Then, do the following
```bash
python3 -m mypy .
python3 -m pytest -v -s tests/unittest/
./tests/integration_test.sh
./tests/regression_test.py
```
### formatting
`mohou` code follows [black](https://github.com/psf/black) standard style. Additionally, we use isort and flake8 to check if the code is following pep standard. Basically, what you have to do for formatting is running 
```bash
./format.sh
```
To running the format command, you need to install 
```bash
pip3 install black isort flake8 autoflake
```
