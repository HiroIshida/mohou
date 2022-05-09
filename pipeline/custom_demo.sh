#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local project_name=$1
    local image_type=$2Image
    local n_pixel=112
    # Try different type of auto encoder
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 4000 -image $image_type -latent 8
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 4000 -image $image_type -latent 12
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name

    python3 -m mohou.script.train_lstm -pn $project_name -n 40000 -aug 9
    python3 -m mohou.script.train_lstm -pn $project_name -n 40000 -aug 9
    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
}

demo_batch pr2_reaching RGB  # replace pr2_reaching with your own project name
# demo_batch pr2_reaching RGBD  # replace pr2_reaching with your own project name
