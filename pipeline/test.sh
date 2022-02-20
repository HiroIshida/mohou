#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example
project_name='pipeline_test'

python3 $example_path/kuka_reaching.py -pn $project_name -n 7
python3 $example_path/train_autoencoder.py -pn $project_name -n 3 -timer-period 1
python3 $example_path/visualize_autoencoder_result.py -pn $project_name
python3 $example_path/train_lstm.py -pn $project_name -valid-ratio 0.5 -n 3
python3 $example_path/visualize_lstm_result.py -pn $project_name
python3 $example_path/visualize_train_history.py -pn $project_name
