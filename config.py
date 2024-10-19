
HOST_IP = "10.11.1.159"

train_machine_info = {
    "10.11.1.159" : ["uisee@10.11.1.159",   "zxcvbnm,.",     "22"],
    "10.9.160.200": ["radmin@10.9.160.200", "uisee@3090",    "22"]
    # "10.0.89.150" : ["test@10.0.89.150",    "test135",      "22"],
    # # "10.9.100.30" : ["radmin@10.9.100.30",  "uisee@2019",   "22"],   # conda 不正常
    # # "10.9.100.31" : ["user@10.9.100.31",    "lp@uisee",     "22"],   # 没有空间了
    # "10.9.100.32" : ["user@10.9.100.32",    "lp@uisee",     "22"],
    # "10.9.100.34" : ["user@10.9.100.34",    "123456",       "22"],
    # # "10.9.100.36" : ["user@10.9.100.36",    "123456",       "22"],   # 看上去很多人会用
}

train_machine_tool_dir = {
    "10.11.1.159" :  "/home/uisee/MainDisk/TrafficLight/yolov7",
    "10.9.160.200":  "/home/radmin/user_data/xiaoqiang/yolov7",
    # "10.0.89.150" :  "/storage/cxq/yolov7",
    # # "10.9.100.30" :  "",
    # # "10.9.100.31" :  "",
    # "10.9.100.32" :  "/sdc/cxq/yolov7",
    # "10.9.100.34" :  "/data3/cxq/yolov7",
    # # "10.9.100.36" :  "",
}

launch_train_template = '''
#!/bin/bash
echo "remove history cache"
rm -r dataset/**/*cache

echo "activate python env"
conda activate yolov7

echo "gen lastly data yaml"
python gen_data_yaml.py $dataset_list

echo "launch tensorboard"
killall tensorboard
tensorboard --logdir runs/train --port 6001 --bind_all &

echo "launch_train"
# python train.py --workers $worker_num --device $device_num --batch-size $batch_size --data data/uisee_data.yaml --img 1280 1280 --epochs $epoch_num --cfg cfg/training/yolov7-tiny-relu.yaml --name $project_name --hyp data/hyp.finetune.yaml --weights $base_model --notest

python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --workers  $worker_num --device  $device_num --sync-bn --batch-size $batch_size --data data/uisee_data.yaml --img 1280 1280 --epochs $epoch_num --cfg cfg/training/yolov7-tiny-relu.yaml --name $project_name --weights $base_model --hyp $train_hyp

killall tensorboard
rm -r $project_name.log
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
UPLOAD_TMP_DIR = "temp"
TEMPLATE_DIR = "template"
DOWNLOAD_RESULT = "download_result"
PROC_DIR = "/dev/shm/train_task_progress.pkl"



def if_not_exist_create(path):
    import os
    if not os.path.exists(path):
        os.mkdir(path)

if_not_exist_create(MODEL_DEPLOY_SAVE_DIR)
if_not_exist_create(TASK_INFO_DIR)
if_not_exist_create(LOG_DIR)
if_not_exist_create(MODEL_REPO_DIR)
if_not_exist_create(DATASET_DIR)
if_not_exist_create(UPLOAD_TMP_DIR)
if_not_exist_create(TEMPLATE_DIR)
if_not_exist_create(DOWNLOAD_RESULT)

