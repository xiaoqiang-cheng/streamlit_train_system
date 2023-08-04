import sys
import subprocess
import os
from config import *
import multiprocessing
from send import send_mail_personal
from utils import *


def push_file(local_path, remote_addr, remote_path, passwd, port):
    command = "sshpass -p %s scp -r -P %s %s %s:%s"%(passwd, port, local_path, remote_addr, remote_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error.decode() == "":
        return True
    else:
        return False

def pull_file(local_path, remote_addr, remote_path, passwd, port):
    command = "sshpass -p %s scp -r -P %s %s:%s %s"%(passwd, port, remote_addr, remote_path, local_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error.decode() == "":
        return True
    else:
        return False

def exec_remote_cmd(ctype, cmd, need_return = False):
    remote_addr, pwd, port = train_machine_info[ctype]
    command = "sshpass -p %s ssh -p%s %s \"%s\""%(pwd, port, remote_addr, cmd)
    SEND_LOG_MSG.info(command)
    if need_return:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return output.decode()


def get_remote_gpus_info(ctype):
    target_dir = train_machine_tool_dir[ctype]
    cmd = "cd %s && ./usertools/get_gpus_info.sh"%(target_dir)
    return exec_remote_cmd(ctype, cmd, need_return = True)

import time

def rsync_remote_dir(passwd, local_path, remote_addr, remote_path, port):

    command = "sshpass -p %s rsync -avz --progress -e 'ssh -p %s' %s/ %s:%s"%(passwd, port, local_path, remote_addr, remote_path)
    print(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    SEND_LOG_MSG.info(output.decode())

def data_sync(remote_ip):
    remote_addr, passwd, port = train_machine_info[remote_ip]
    local_path = "dataset"
    remote_path = os.path.join(train_machine_tool_dir[remote_ip], local_path)
    rsync_remote_dir(passwd, local_path, remote_addr, remote_path, port)

def launch_train():
    time.sleep(10)

def model_deploy():
    time.sleep(10)

def send_email():
    time.sleep(10)

def main(email, remote_ip, project_name):
    # provide double exec
    if os.path.exists("train_task_progressremote_ip.pkl"):
        return

    train_task_progress = {}

    train_task_progress["task_progress"] = 0
    serialize_data(train_task_progress, "train_task_progress.pkl")

    # 数据同步
    train_task_progress["task_progress"] = 1
    SEND_LOG_MSG.info("ready to data sync.")
    data_sync(remote_ip)
    serialize_data(train_task_progress, "train_task_progress.pkl")

    # 开始训练
    train_task_progress["task_progress"] = 2
    SEND_LOG_MSG.info("ready to launch train")
    launch_train()
    serialize_data(train_task_progress, "train_task_progress.pkl")

    #量化部署
    train_task_progress["task_progress"] = 3
    SEND_LOG_MSG.info("ready to deploy")
    model_deploy()
    serialize_data(train_task_progress, "train_task_progress.pkl")

    #发送邮件
    train_task_progress["task_progress"] = 4
    SEND_LOG_MSG.info("ready to send email")
    send_email()
    serialize_data(train_task_progress, "train_task_progress.pkl")
    train_task_progress["task_progress"] = -1
    serialize_data(train_task_progress, "train_task_progress.pkl")
    SEND_LOG_MSG.info("pipe done")


if __name__=="__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default="xiaoqiang.cheng@uisee.com")
    parser.add_argument('--remote-ip', type=str, default="10.0.93.231")
    parser.add_argument('--project', type=str, default="uisee")
    args = parser.parse_args()



    main(args.email, args.remote_ip, args.project)