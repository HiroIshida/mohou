#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=pybullet_$1

    python3 $example_path/pybullet/create_dataset.py -pn $project_name -n 60
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 1000 -image $image_type
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name -image $image_type
    python3 -m mohou.script.train_lstm -pn $project_name -n 6000 -image $image_type
    python3 -m mohou.script.visualize_lstm_result -pn $project_name -image $image_type
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/pybullet/simulate_feedback.py -pn $project_name
}

demo_batch RGB
demo_batch Depth
demo_batch RGBD
