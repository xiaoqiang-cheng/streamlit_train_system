
train_machine_info = {
    "10.0.93.231" : ["uisee@10.0.93.231",   "8888",         "22"],
    "10.0.89.150" : ["test@10.0.89.150",    "test135",      "22"],
    "10.9.100.30" : ["radmin@10.9.100.30",  "uisee@2019",   "22"],
    "10.9.100.31" : ["user@10.9.100.31",    "lp@uisee",     "22"],
    "10.9.100.32" : ["user@10.9.100.32",    "lp@uisee",     "22"],
    "10.9.100.34" : ["user@10.9.100.34",    "123456",       "22"],
    "10.9.100.36" : ["user@10.9.100.36",    "123456",       "22"],
    "10.9.160.200": ["radmin@10.9.160.200", "uisee@123",    "22"]
}

train_machine_tool_dir = {
    "10.0.93.231" :  "/home/uisee/MainDisk/TrafficLight/yolov7",
    "10.0.89.150" :  "",
    "10.9.100.30" :  "",
    "10.9.100.31" :  "",
    "10.9.100.32" :  "",
    "10.9.100.34" :  "",
    "10.9.100.36" :  "",
    "10.9.160.200":  "",
}

launch_train_template = '''
#!/bin/bash
rm -r dataset/**/*cache

source ~/.bashrc
source ~/.zshrc

set -e
conda activate yolov7
tensorboard --logdir runs/train --port 6001 --bind_all
python gen_data_yaml.py
python train.py --workers $worker_num --device $device_num --batch-size $batch_size --data data/uisee_data.yaml --img 1280 1280 --epochs $epoch_num --cfg cfg/training/yolov7-tiny-relu.yaml --name $project_name --hyp data/hyp.finetune.yaml --weights $base_model
killall tensorboard
'''

machine_info = {
    "xavier" : ["worker@10.0.93.199", "uisee", "22"],
    "tx2" : ["worker@10.0.93.200", "uisee", "22"],
    "1080ti" : ["root@10.9.100.32", "123456", "9975"],
    "J5" : [],
    "8155" : []
}

TRT4_INDEX = 0
TRT8_INDEX = 1

engine_type_str = ["trt4", "trt8"]

machine_tool_dir = {
    "xavier"    : ["",
                    "/home/worker/xiaoqiang/TensorRT/", ],
    "tx2"       : ["/media/worker/Samsung_T5/cxq/tensorrt_deploy/tensorrtinference/release",
                    ],
    "1080ti"    : ["/root/tensorrtinference/release",
                    "/root/TensorRT-8.0.1.6"],
    "J5"        : [],
    "8155"      : []
}

supply_choice = {
    "1080ti" : ["trt4", "trt8"],
    "tx2"    : ["trt4",],
    "xavier" : ["trt8"],
    "J5" : ["todo"],
    "8155" :  ["todo"]
}



MODEL_DEPLOY_SAVE_DIR = "model_deploy_save"
TASK_INFO_DIR = "task_info"
LOG_DIR = "log"
MODEL_REPO_DIR = "model_repo"
DATASET_DIR = "dataset"



def if_not_exist_create(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

if_not_exist_create(MODEL_DEPLOY_SAVE_DIR)
if_not_exist_create(TASK_INFO_DIR)
if_not_exist_create(LOG_DIR)
if_not_exist_create(MODEL_REPO_DIR)
if_not_exist_create(DATASET_DIR)
