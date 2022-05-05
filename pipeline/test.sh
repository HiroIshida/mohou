#/bin/bash
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
    # TODO(HiroIshida) bit dirty
    python3 $example_path/train_autoencoder.py -pn $project_name -n 2 -image $image_type $vae_option

    cp ~/.mohou/$project_name/MultiEpisodeChunk.pkl ~/.mohou/$project_name/MultiEpisodeChunk-auxiliary.pkl
    if [ $image_type = RGBImage ]; then  # once is enough
        python3 $example_path/train_autoencoder.py -pn $project_name -n 2 -image $image_type --warm $vae_option
        python3 $example_path/train_autoencoder.py -pn $project_name -n 2 -image $image_type --aux $vae_option
    fi
    python3 $example_path/visualize_autoencoder_result.py -pn $project_name -n 2

    # train lstm two times
    python3 $example_path/train_lstm.py -pn $project_name -valid-ratio 0.5 -n 2
    if [ $image_type RGBImage ]; then  # once is enough
        python3 $example_path/train_lstm.py -pn $project_name -valid-ratio 0.5 -n 2 --warm
    fi

    python3 $example_path/visualize_lstm_result.py -pn $project_name -n 5
    python3 $example_path/visualize_train_history.py -pn $project_name
    python3 $example_path/kuka_reaching.py -pn $project_name --feedback
}

test_batch RGB true
test_batch RGB false
test_batch Depth false
test_batch RGBD false
