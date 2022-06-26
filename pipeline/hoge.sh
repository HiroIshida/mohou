#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local task_name=$2
    local camera_name=$3
    local project_name=rlbench_demo_$task_name
    #python3 $example_path/rlbench/create_dataset.py -pn $project_name -n 65 -tn $task_name -resol 112 -cn $camera_name

    #python3 -m mohou.script.train_autoencoder -pn $project_name -n 1500 -image $image_type --vae
    #python3 -m mohou.script.visualize_autoencoder_result -pn $project_name
    #python3 -m mohou.script.train_lstm -pn $project_name -n 20000 -aug 9
 
    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
    #python3 $example_path/rlbench/simulate_feedback.py -pn $project_name -n 250 -tn $task_name
}

demo_batch RGB StackChairs front
#demo_batch RGB SweepToDustpan front
#demo_batch RGB WipeDesk front
#demo_batch RGB ModifiedEmptyDishwasher left_shoulder
