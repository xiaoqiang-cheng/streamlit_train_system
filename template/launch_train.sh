
#!/bin/bash
rm -r dataset/**/*cache

source ~/.bashrc
source ~/.zshrc

set -e
conda activate yolov7
tensorboard --logdir runs/train --port 6001 --bind_all
python gen_data_yaml.py
python train.py --workers 32 --device 0,1,2 --batch-size 32 --data data/uisee_data.yaml --img 1280 1280 --epochs 32 --cfg cfg/training/yolov7-tiny-relu.yaml --name 32 --hyp data/hyp.finetune.yaml --weights base_model.pt
