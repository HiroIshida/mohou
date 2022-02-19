#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example
python3 $example_path/kuka_reaching.py -n 2
python3 $example_path/train_autoencoder.py
python3 $example_path/train_lstm.py
