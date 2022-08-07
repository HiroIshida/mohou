#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local n_teaching=$2
    local task_name=$3
    local camera_name=$4
    local resolution=$5
    local n_process=$6  # if n_process = 0, then number of process is automatically determined

    local project_name=rlbench_demo_$task_name

    python3 $example_path/rlbench/create_dataset.py \
        -pn $project_name \
        -n $n_teaching \
        -tn $task_name \
        -cn $camera_name \
        -p $n_process \
        -resol $resolution

    python3 -m mohou.script.train_autoencoder -pn $project_name -n 2000 -image $image_type --vae
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name
    python3 -m mohou.script.train_lstm -pn $project_name -n 20000 -aug 9
 
    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/rlbench/simulate_feedback.py -pn $project_name -n 250
}

demo_batch RGBD 45 CloseDrawer overhead 112 0
