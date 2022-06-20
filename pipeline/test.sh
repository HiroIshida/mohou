#/bin/bash
set -e

base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function test_batch {
    local image_type=$1Image
    local use_vae=$2
    local project_name=_pipeline_test_$1

    local vae_option=""
    if [ $use_vae = true ]; then
        project_name="${project_name}_vae"
        vae_option="--vae"
    fi
    python3 $example_path/kuka_reaching.py -pn $project_name -n 7
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 2 -image $image_type $vae_option

    if [ $image_type = RGBImage ]; then  # once is enough
        python3 -m mohou.script.train_autoencoder -pn $project_name -n 2 -image $image_type --warm $vae_option
    fi
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name -n 2

    # train lstm two times
    python3 -m mohou.script.train_lstm -pn $project_name -valid-ratio 0.5 -n 2
    if [ $image_type RGBImage ]; then  # once is enough
        python3 -m mohou.script.train_lstm -pn $project_name -valid-ratio 0.5 -n 2 --warm
    fi

    python3 -m mohou.script.visualize_lstm_result -pn $project_name -n 5
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/kuka_reaching.py -pn $project_name --feedback
}

test_batch RGB true
test_batch RGB false
test_batch Depth false
test_batch RGBD false
