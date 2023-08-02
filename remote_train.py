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