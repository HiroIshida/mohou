#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

python3 $example_path/kuka_reaching.py -n 100
python3 $example_path/train_autoencoder.py -n 3000
python3 $example_path/train_lstm.py -n 4000
