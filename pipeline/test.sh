#/bin/bash
set -e

base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function test_batch {
    local image_type=$1Image
    local project_name=_pipeline_test_$1
    local use_vae=$2
    local test_warm_train=$3
    local use_context=$4
    local use_chimera=$5  # train chimera (ae + lstm) instead of lstm

    local vae_option=""
    if [ $use_vae = true ]; then
        project_name="${project_name}_vae"
        vae_option="--vae"
    fi

    local context_flag=""
    if [ $use_context = true ]; then
        project_name="${project_name}_with_context"
        context_flag="--use_image_context"
    fi

    # echo test condition
    echo "==== test condition === "
    echo "image_type: $1"
    echo "use_vae: $2"
    echo "test_warm_train: $3"
    echo "use_context: $4"
    echo "use_chimera: $5"

    python3 $example_path/kuka_reaching.py -pn $project_name -n 7
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 2 -image $image_type $vae_option

    if [ $test_warm_train = true ]; then
        python3 -m mohou.script.train_autoencoder -pn $project_name -n 2 -image $image_type --warm $vae_option
    fi
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name -n 2

    if [ $use_chimera = true ]; then
        echo "train chimera"
        python3 -m mohou.script.train_chimera -pn $project_name -valid-ratio 0.5 -n 2
        python3 -m mohou.script.visualize_lstm_result -pn $project_name -n 5 --chimera
        python3 -m mohou.script.visualize_train_history -pn $project_name
        # TODO(HiroIshida): test feedback
    else  # train lstm
        # train lstm two times
        python3 -m mohou.script.train_lstm -pn $project_name -valid-ratio 0.5 -n 2 $context_flag
        if [ $test_warm_train = true ]; then
            python3 -m mohou.script.train_lstm -pn $project_name -valid-ratio 0.5 -n 2 --warm $context_flag
        fi
        python3 -m mohou.script.visualize_lstm_result -pn $project_name -n 5
        python3 -m mohou.script.visualize_train_history -pn $project_name
        python3 $example_path/kuka_reaching.py -pn $project_name --feedback
    fi

    rm -rf ~/.mohou/$project_name
}

test_batch RGB true true false # test warm train
test_batch RGB false false true # test using context
test_batch RGB false false false true # test chimera
test_batch RGB false false false
test_batch Depth false false false
test_batch RGBD false false false
