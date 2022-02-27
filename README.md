## mohou [![CI](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml) [![PypI Auto Release](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions/workflows/release.yaml) [![PyPI version](https://badge.fury.io/py/mohou.svg)](https://pypi.org/project/mohou/)

This package implements imitation learning trainer and executor using pytorch. Currently the library targets autoencoder-lstm-type behavior cloning.

## Tutorial demo

<img src="https://user-images.githubusercontent.com/38597814/155882282-f40af02b-99aa-41b3-bd43-fe7b7d0c2d96.gif" width="30%" /><img src="https://user-images.githubusercontent.com/38597814/155882252-5739fa16-baf7-4a26-b88f-24e106ea0dd1.gif" width="30%" />

left: teaching sample (`~/.mohou/pipeline_test_RGBD/sample.gif`)
right: testing sample (`~/.mohou/pipeline_test_RGBD/feedback_simulation.gif`)

Running [`pipeline/demo.sh`](/pipeline/demo.sh) is a good first step. Note that a key concept of this library is a "project", where all data, learned models, result visualizations and logs are stored in a project directory `~/.mohou/{project_name}`. 

For example, after running `demo_batch RGBD` in [`pipeline/demo.sh`](/pipeline/demo.sh), we can confirm that following directly sturecture under the corresponding project directory.
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

<details>
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
Typical data collection code looks like the following, where `AngleVector`, `RGBImage` and `DepthImage` is stored here but any combination of `ElementBase`'s subtype (see mohou/types.py) can be used.
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
        rgb = RGBImage(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        depth = DepthImage(np.random.randn(224, 224, 1))
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
*under construction*

## Define custom element type
*under construction*
