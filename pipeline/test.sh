#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

python3 $example_path/kuka_reaching.py -n 2
python3 $example_path/train_autoencoder.py -n 3 -timer-period 1
python3 $example_path/train_lstm.py -valid-ratio 0.5 -n 3
