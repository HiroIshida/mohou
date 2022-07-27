## mohou [![CI](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml) [![PypI Auto Release](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml) [![PyPI version](https://badge.fury.io/py/mohou.svg)](https://pypi.org/project/mohou/)

This package implements conv-autoencoder based visuo-motor imitation learning using pytorch. This package focuses on extensibility. One can define custom data types (vector, image and composite-image) and also map/inverse from these custom types to feature vector that fed into the LSTM. Alongside the imitation learning framework, this package provides two demo using pybullet and rlbench. 

Please try [kinematic pybullet demo](/pipeline/pybullet_demo.sh) and [dynamic rlbench demo](/pipeline/rlbench_demo.sh). The results of demo is available on [google drive](https://drive.google.com/drive/folders/1wzEk4u3B0LvvErki3E-zifUqgtZoqVxF).

<img src="https://user-images.githubusercontent.com/38597814/160765355-b1d490a9-fb9b-40da-b1fb-d9eae476fda0.gif" width="50%" />

one of result of applying this framework to rlbench's task

## Instllation
```
git clone https://github.com/HiroIshida/mohou.git
cd mohou
pip install -e . 
```
After this pip install you are ready to start [pybullet demo](/pipeline/pybullet_demo.sh)! We also provide [rlbench_demo](/pipeline/rlbench_demo.sh). As for rlbench demo, additional installation step of pyrep and rlbench is required.  See https://github.com/stepjam/RLBench for the detail.


## Introduction
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


## Contribution
When you make a new PR, one need to check that the tests passed and formatting is correct.

### testing
The test for `mohou` software is 3 steps: 1) static type check by mypy, 2) unit test by pytest and 3) integration test.
To running these test, you need install `mypy` and `pytest` by
```
pip3 install pytest mypy  # or use --user option 
```
Then, do the following
```bash
mypy .
pytest -v -s
bash ./pipeline/test.sh
```
### formatting
`mohou` code follows [black](https://github.com/psf/black) standard style. Additionally, we use isort and flake8 to check if the code is following pep standard. Basically, what you have to do for formatting is running 
```
./format.sh
```
(If this command is not sufficient to pass the lint test in CI, please send me a PR)

Note that to run the command you need to install packages
```
pip3 install black isort flake8 autoflake
```


### Data collection
Typical data collection code looks like the following, where `AngleVector`, `RGBImage` and `DepthImage` are stored here but any combination of `ElementBase`'s subtype (see mohou/types.py) such as `AngleVector` plus `RGBImage` or `AngleVector` plut `DepthImage` can be used. You can also define custom type see [this](https://github.com/HiroIshida/mohou#define-custom-element-type).

```python
import numpy as np

from mohou.types import (
    AngleVector,
    DepthImage,
    ElementSequence,
    EpisodeData,
    MultiEpisodeChunk,
    RGBImage,
)


def create_episode_data():
    n_step = 100
    rgb_seq = ElementSequence()
    depth_seq = ElementSequence()
    av_seq = ElementSequence()
    for _ in range(n_step):
        rgb = RGBImage(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )  # replace this by actual data
        depth = DepthImage(np.random.randn(224, 224, 1))  # replace this by actual data
        av = AngleVector(np.random.randn(7))  # replace this by actual data

        rgb_seq.append(rgb)
        depth_seq.append(depth)
        av_seq.append(av)
    return EpisodeData.from_seq_list([rgb_seq, depth_seq, av_seq])


n_episode = 20
data_list = [create_episode_data() for _ in range(n_episode)]
chunk = MultiEpisodeChunk.from_data_list(data_list)
chunk.dump("dummy_project")  # dumps to ~/.mohou/dummy_project/MultiEpisodeChunk.pkl
```

### Execution
Typical code for execution using learned propgatos is as follows. Note that type-hinting here is just for explanation and not necessarily required.
```python
from mohou.default import create_default_propagator
from mohou.propagator import Propagator
from mohou.types import AngleVector, ElementDict, RGBImage

# Please change project_name and n_angle_vector
propagator: Propagator = create_default_propagator("your_project_name")

while True:
    # Observation using real/simulated robot
    rgb: RGBImage = obtain_rgb_image()  # define by yourself
    av: AngleVector = obtain_angle_vector()  # define by yourself
    elem_dict = ElementDict((rgb, av))

    propagator.feed(elem_dict)

    # If your fed elem_dict contains RGBImage and AngleVector, then propagated
    # elem_dict_pred also has RGBImage and AngleVector
    elem_dict_pred: ElementDict = propagator.predict(n_prop=1)[0]

    # Get specific element
    rgb_pred = elem_dict_pred[AngleVector]
    av_pred = elem_dict_pred[RGBImage]

    # send command
    send_next_angle_vector(av_pred)  # define by yourself
```

## Define custom element type
The following figure show the type hierarchy. In this framework, only the leaf types (filled by grey) can be instantiated. In most case, users would create custom type by inheriting from either `CompositeImageBase`, or `VectorBase` or `PrimitiveImageBase`. For the detail, please refere to [`mohou/types.py`](/mohou/types.py) for how the built-in concrete types such as `RGBDImage`, `RGBImage` and `AngleVector` are defined.

<img src="https://user-images.githubusercontent.com/38597814/156465428-35a54445-3c2b-498d-8983-23550d77415c.png" width="60%" />

## Define custom embedder
`Embedder` in this framework is to embed `element: ElementBase` to 1-dim `np.ndarray`. For example, the built-in embedder `ImageEmbedder` equipped with a map from an image to a vector and a map from a feature vector to an image.

You could define your custom embedder by inheriting `EmbedderBase` and define both methods:
- `_forward_impl(self, inp: ElementT) -> np.ndarray`
- `_backward_impl(self, inp: np.ndarray) -> ElementT`

For example, in the demo we create `ImageEmbedder` from `AutoEncoder` instance, but you could use PCA or other dimension reduction methods.
