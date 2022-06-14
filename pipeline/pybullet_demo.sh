#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=pybullet_reaching_$1
    local n_pixel=$2
    python3 $example_path/kuka_reaching.py -pn $project_name -n 60 -m $2
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 1500 -image $image_type --vae -aug 0
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name

    python3 -m mohou.script.train_lstm -pn $project_name -n 20000 -aug 9

    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/kuka_reaching.py -pn $project_name --feedback -m $2
}

demo_batch RGB 224  # must be 112 or 224
# demo_batch Depth 224 # comment out this if you want
# demo_batch RGBD 224 # comment out this if you want
