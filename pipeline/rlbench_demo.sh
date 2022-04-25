#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=rlbench_demo
     python3 $example_path/rlbench/create_dataset.py -pn $project_name -n 105
     python3 $example_path/train_autoencoder.py -pn $project_name -n 2000 -image $image_type
     python3 $example_path/visualize_autoencoder_result.py -pn $project_name
 
     for i in $(seq 2); do
         python3 $example_path/train_lstm.py -pn $project_name -n 20000 -aug 4
     done
 
    python3 $example_path/visualize_lstm_result.py -pn $project_name
    python3 $example_path/visualize_train_history.py -pn $project_name
    python3 $example_path/rlbench/simulate_feedback.py -pn $project_name -n 250
}

demo_batch RGBD
