### mohou [![CI](https://github.com/HiroIshida/mohou/actions/workflows/test.yaml/badge.svg)](https://github.com/HiroIshida/mohou/actions)

This package implements imitation learning trainer and executor using pytorch. Currently the library targets autoencoder-lstm-type behavior cloning.

### Demo
see [`pipeline/demo.sh`](/pipeline/demo.sh) and corresponding scripts in `example` are good tutorials.

- [`example/kuka_reaching.py`](/example/kuka_reaching.py) creates set of example of trajectory achieving inverse kinematics (gif below). 
<img src="https://drive.google.com/uc?export=view&id=1uL4eEbZ8OmbdBQKDox75aUzqHDUZavP2" width="200px">

- Then we train autoencoder by [`example/train_autoencoder.py`](/example/train_autoencoder.py) and lstm by [`example/train_lstm.py`](/example/train_lstm.py) in order. Note that in the latter step, autoencoder is need to be already trained. 

- Trained lstm's n-ahead prediction is visualized and saved in `~/.mohou/kuka_reaching/lstm_result.py` as a gif (gif below)

<img src="https://drive.google.com/uc?export=view&id=1AV1iQje-a9WyTi9p9PXRsXbXuqO5ARVl" width="300px">