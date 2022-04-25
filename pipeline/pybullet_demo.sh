#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=pybullet_reaching_$1
    local n_pixel=$2
    python3 $example_path/kuka_reaching.py -pn $project_name -n 60 -m $2
    python3 $example_path/train_autoencoder.py -pn $project_name -n 1000 -image $image_type
    python3 $example_path/visualize_autoencoder_result.py -pn $project_name

    for i in $(seq 2); do
        python3 $example_path/train_lstm.py -pn $project_name -n 10000 -aug 4
    done

    python3 $example_path/visualize_lstm_result.py -pn $project_name
    python3 $example_path/visualize_train_history.py -pn $project_name
    python3 $example_path/kuka_reaching.py -pn $project_name --feedback -m $2
}

demo_batch RGB 224  # must be 112 or 224
# demo_batch Depth 224 # comment out this if you want
# demo_batch RGBD 224 # comment out this if you want
