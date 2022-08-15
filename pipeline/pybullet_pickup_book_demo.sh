#/bin/bash
base_path=$(dirname "$(realpath $0)")
example_path=$base_path/../example

function demo_batch {
    local camera=$1
    local n_pixel=$2
    local project_name=panda_pickup_book-$camera
    python3 $example_path/pybullet_complex_task/pickup_book.py -pn $project_name -n 85 --headless -camera $camera
    python3 -m mohou.script.train_autoencoder -pn $project_name -n 1500 -image RGBImage --vae -aug 0
    python3 -m mohou.script.visualize_autoencoder_result -pn $project_name

    python3 -m mohou.script.train_lstm -pn $project_name -n 20000 -aug 9

    python3 -m mohou.script.visualize_lstm_result -pn $project_name
    python3 -m mohou.script.visualize_train_history -pn $project_name
    python3 $example_path/pybullet_complex_task/pickup_book.py -pn $project_name --feedback --headless -camera $camera
}

demo_batch lefttop 224  # must be 112 or 224
#demo_batch front 224  # must be 112 or 224
