#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function test_batch {
    local image_type=$1Image
    local project_name=_pipeline_test_$1
    python3 $example_path/pybullet/create_dataset.py -pn $project_name -n 7
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 2 -timer-period 1 -image $image_type
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name -image $image_type -n 2
    python3 -m mohou.script.train_lstm -pn $project_name -valid-ratio 0.5 -n 2 -image $image_type
    python3 -m mohou.script.visualize_lstm_result -pn $project_name -image $image_type -n 2
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/pybullet/simulate_feedback.py -pn $project_name
}

test_batch RGB
test_batch Depth
test_batch RGBD
