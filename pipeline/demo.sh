#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

python3 $example_path/kuka_reaching.py -n 60
python3 $example_path/train_autoencoder.py -n 1000
python3 $example_path/visualize_autoencoder_result.py
python3 $example_path/train_lstm.py -n 6000
python3 $example_path/visualize_lstm_result.py
python3 $example_path/visualize_train_history.py
