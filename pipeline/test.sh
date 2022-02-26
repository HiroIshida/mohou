#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function test_batch {
    local image_type=$1Image
    local project_name=pipeline_test_$1
    python3 $example_path/kuka_reaching.py -pn $project_name -n 7
    python3 $example_path/train_autoencoder.py -pn $project_name -n 2 -timer-period 1 -image $image_type
    python3 $example_path/visualize_autoencoder_result.py -pn $project_name -image $image_type -n 2
    python3 $example_path/train_lstm.py -pn $project_name -valid-ratio 0.5 -n 2 -image $image_type
    python3 $example_path/visualize_lstm_result.py -pn $project_name -image $image_type -n 2
    python3 $example_path/visualize_train_history.py -pn $project_name
}

test_batch RGB
test_batch Depth
test_batch RGBD
