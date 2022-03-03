## mohou [![CI](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml) [![PypI Auto Release](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml) [![PyPI version](https://badge.fury.io/py/mohou.svg)](https://pypi.org/project/mohou/)

This package implements imitation learning trainer and executor using pytorch. Currently the library targets autoencoder-lstm-type behavior cloning.

## Tutorial demo (vision-based reaching task)

<img src="https://user-images.githubusercontent.com/38597814/155882282-f40af02b-99aa-41b3-bd43-fe7b7d0c2d96.gif" width="30%" /><img src="https://user-images.githubusercontent.com/38597814/155882252-5739fa16-baf7-4a26-b88f-24e106ea0dd1.gif" width="30%" />

left: teaching sample (`~/.mohou/pipeline_test_RGBD/sample.gif`)

right: testing sample (`~/.mohou/pipeline_test_RGBD/feedback_simulation.gif`)

Running [`pipeline/demo.sh`](/pipeline/demo.sh) is a good first step. Note that GPU is necessary in practice, though the software works on cpu. Result of `pipeline/demo.sh` is available in [google drive](https://drive.google.com/drive/u/0/folders/1ngozVlXah1eBlduydF5uJCA7tRg1EiRb). 

A key concept of this library is a "project", where all data, learned models, result visualizations and logs are stored in a project directory `~/.mohou/{project_name}`. For example, after running `demo_batch RGBD` in [`pipeline/demo.sh`](/pipeline/demo.sh), we can confirm that following directly sturecture under the corresponding project directory.
```
h-ishida@ccddbeeedc93:~$ tree ~/.mohou/pipeline_test_RGBD/
/home/h-ishida/.mohou/pipeline_test_RGBD/
├── MultiEpisodeChunk.pkl
├── TrainCache-AutoEncoder.pkl
├── TrainCache-LSTM.pkl
├── autoencoder_result
│   ├── result0.png
│   ├── result1.png
│   ├── result2.png
│   ├── result3.png
│   └── result4.png
├── log
│   ├── autoencoder_20220226224448.log
│   ├── autoencoder_20220226225242.log
│   ├── latest_autoencoder.log -> /home/h-ishida/.mohou/pipeline_test_RGBD/log/autoencoder_20220226225242.log
│   ├── latest_lstm.log -> /home/h-ishida/.mohou/pipeline_test_RGBD/log/lstm_20220227022128.log
│   └── lstm_20220227022128.log
├── lstm_result
│   └── result.gif
├── sample.gif
└── train_history
    ├── TrainCache-AutoEncoder.pkl.png
    └── TrainCache-LSTM.pkl.png
```

<details open>
<summary> detail of each component of demo.sh </summary>

- `kuka_reaching.py` creates `MultiEpisodeChunk.pkl` which consists of `n` sample trajectories that reaches to the box in the image (stored in `~/.mohou/{project_name}/). The datachunk consists of sequences of `RGBImage` and `DepthImage` and `AngleVector`. Also, one of the trajectory image in the chunk is visualized as `~/.mohou/{project_name}/sample.gif`.

- `train_autoencoder.py` trains an autoencoder of `$image_type`. $image_type can either be `RGBImange`, `DepthImage` or `RGBDImage`. The train cache is stored as `~/.mohou/{project_name}/TrainCache-AutoEncoder.pkl`.

- `visualize_autoencoder_result.py` visualize the comparison of original and reconstructed image by the autoencoder (stored in `~/.mohou/{project_name}/autoencoder_result/)`. This visualization is useful for debugging/tunning, especially to determine the train epoch of autoencoder if needed.

- `train_lstm.py` trains and lstm that propagate vectors concated by feature vector compressed by the trained autoencoder and `AngleVector`. Note that `train_autoencoder.py` must be run beforehand. The train cache is stored as `~/.mohou/{project_name}/TrainCache-LSTM.pkl`.

- `visualize_lstm_result.py` visualizes the `n` step prediction given 10 images, which can be used for debugging/tuning or determining the good training epoch of the lstm training. The gif file is stored as `~/.mohou/{project_name}/lstm_result/result.gif`
<img src="https://user-images.githubusercontent.com/38597814/155882256-39a55b42-9973-4a66-94ee-a08df273c1cf.gif" width="30%" />

- `visualize_train_history.py` visualizes the training history (test and validation loss) for all train caches in the project directory. The figures will be stored in `~/.mohou/{project_name}/train_history/`

- `kuka_reaching.py --fedback` simualte the visuo-motor reaching task in the simulator using trained autoencoder and lstm. The visualization of the simulation is stored as `~/.mohou/{project_name}/feedback_simulation.gif`.

Also note that logs by `train_autoencoder.py` and `train_lstm.py` will be stored in `~/.mohou/{project_name}/log/`.
</details>


## Applying to your own project
Besides parameter/training epoch tuning, to applyig this software to your own project you must replace 
- `kuka_reaching.py` (data collection) by your own data creation program using real robot (or simulated model)
- `kuka_reaching.py --feedback` (execution) by real robot execution program
among the `pipeline/demo.sh`.

### Data collection
Typical data collection code looks like the following, where `AngleVector`, `RGBImage` and `DepthImage` are stored here but any combination of `ElementBase`'s subtype (see mohou/types.py) such as `AngleVector` plus `RGBImage` or `AngleVector` plut `DepthImage` can be used. You can also define custom type see [this](https://github.com/HiroIshida/mohou#define-custom-element-type).
```python
import numpy as np
from mohou.types import RGBImage, DepthImage, AngleVector
from mohou.types import ElementSequence, EpisodeData, MultiEpisodeChunk

def create_episode_data():
    n_step = 100
    rgb_seq = ElementSequence[RGBImage]()
    depth_seq = ElementSequence[DepthImage]()
    av_seq = ElementSequence[AngleVector]()
    for _ in range(n_step):
        rgb = RGBImage(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))  # replace this by actual data
        depth = DepthImage(np.random.randn(224, 224, 1))  # replace this by actual data
        av = AngleVector(np.random.randn(7))  # replace this by actual data

        rgb_seq.append(rgb)
        depth_seq.append(depth)
        av_seq.append(av)
    return EpisodeData((rgb_seq, depth_seq, av_seq))

n_episode = 20
chunk = MultiEpisodeChunk([create_episode_data() for _ in range(n_episode)])
chunk.dump('dummy_project')  # dumps to ~/.mohou/dummy_project/MultiEpisodeChunk.pkl
```

### Execution
Typical code for execution using learned propgatos is as follows. Note that type-hinting here is just for explanation and not necessarily required.
```python
from mohou.types import ElementDict, RGBImage, AngleVector
from mohou.propagator import Propagator, create_default_propagator

# Please change project_name and n_angle_vector 
propagator: Propagator = create_default_propagator('your_project_name', n_angle_vector=7)

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
