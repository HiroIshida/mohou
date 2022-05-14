#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=rlbench_demo
    python3 $example_path/rlbench/create_dataset.py -pn $project_name -n 85
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 2000 -image $image_type --vae
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name
    python3 -m mohou.script.train_lstm -pn $project_name -n 20000 -aug 9
 
    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/rlbench/simulate_feedback.py -pn $project_name -n 250
}

demo_batch RGBD
