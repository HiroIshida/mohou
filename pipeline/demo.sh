#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local image_type=$1Image
    local project_name=pipeline_test_$1
    python3 $example_path/kuka_reaching.py -pn $project_name -n 60
    python3 $example_path/train_autoencoder.py -pn $project_name -n 1000 -image $image_type
    python3 $example_path/visualize_autoencoder_result.py -pn $project_name -image $image_type
    python3 $example_path/train_lstm.py -pn $project_name -n 6000 -image $image_type
    python3 $example_path/visualize_lstm_result.py -pn $project_name -image $image_type
    python3 $example_path/visualize_train_history.py -pn $project_name
}

demo_batch RGB
demo_batch Depth
demo_batch RGBD
