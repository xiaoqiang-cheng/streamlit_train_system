import sys
import subprocess
import os
from config import *
import multiprocessing
from send import send_mail_personal
from utils import *
from remote_cvt_model import engine_cvt_pipeline
import time

def push_file(local_path, remote_addr, remote_path, passwd, port):
    command = "sshpass -p %s scp -r -o StrictHostKeyChecking=no -P %s %s %s:%s"%(passwd, port, local_path, remote_addr, remote_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error.decode() == "":
        return True
    else:
        return False

def pull_file(local_path, remote_addr, remote_path, passwd, port):
    command = "sshpass -p %s scp -r -o StrictHostKeyChecking=no -P %s %s:%s %s"%(passwd, port, remote_addr, remote_path, local_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if error.decode() == "":
        return True
    else:
        return False

def exec_remote_cmd(ctype, cmd, need_return = False):
    remote_addr, pwd, port = train_machine_info[ctype]
    command = "sshpass -p %s ssh -o StrictHostKeyChecking=no -p%s %s \'%s\'"%(pwd, port, remote_addr, cmd)
    SEND_LOG_MSG.info(command)
    if need_return:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        process = subprocess.Popen(command, shell=True)
    output, error = process.communicate()

    if need_return:
        SEND_LOG_MSG.info(output.decode())
        return output.decode()
    else:
        pass

def check_remote_file_exist(ctype, file_path):
    remote_addr, pwd, port = train_machine_info[ctype]
    command = "sshpass -p %s ssh -o StrictHostKeyChecking=no -p%s %s \"%s\""%(pwd, port, remote_addr, "ls %s"%file_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error.decode() == "":
        return True
    else:
        return False


def get_remote_gpus_info(ctype):
    target_dir = train_machine_tool_dir[ctype]
    cmd = "cd %s && bash -i ./usertools/get_gpus_info.sh 2>/dev/null"%(target_dir)
    return exec_remote_cmd(ctype, cmd, need_return = True)


def rsync_remote_dir(passwd, local_path, remote_addr, remote_path, port):
    command = "sshpass -p %s rsync -avz --progress -e 'ssh -o StrictHostKeyChecking=no -p %s' %s/ %s:%s"%(passwd, port, local_path, remote_addr, remote_path)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    SEND_LOG_MSG.info(output.decode())
    if error.decode() == "":
        return True
    else:
        return False

def data_sync(remote_ip):
    remote_addr, passwd, port = train_machine_info[remote_ip]
    local_path = "dataset"
    remote_path = os.path.join(train_machine_tool_dir[remote_ip], local_path)
    ret = rsync_remote_dir(passwd, local_path, remote_addr, remote_path, port)
    if not ret:
        SEND_LOG_MSG.error("sync data error!")
        return False
    return True

def launch_train(remote_ip, base_model, project_name):
    remote_addr, passwd, port = train_machine_info[remote_ip]
    remote_path = train_machine_tool_dir[remote_ip]

    if base_model != '':
        local_base_model_path = os.path.join(MODEL_REPO_DIR, base_model, "best.pt")
        remote_base_model_path = os.path.join(remote_path, MODEL_REPO_DIR, base_model + ".pt")

        SEND_LOG_MSG.info("copy base model to remote end")
        ret = push_file(local_base_model_path, remote_addr, remote_base_model_path, passwd, port)
        if not  ret:
            SEND_LOG_MSG.error("copy Model ERROR")
            return False

    # copy train script to remote end
    SEND_LOG_MSG.info("copy launch_train.sh to remote end")
    ret = push_file(os.path.join(TEMPLATE_DIR, "launch_train.sh"), remote_addr, remote_path, passwd, port)
    if not  ret:
        SEND_LOG_MSG.error("copy Model ERROR")
        return False

    tmux_project_name = "traffic_light_%s"%project_name
    train_log_name = project_name + ".log"
    cmd = "cd %s && bash -i launch_train.sh 2>&1 | tee %s"%(remote_path, train_log_name)
    tmux_session_cmd = f'tmux has-session -t {tmux_project_name} 2>/dev/null || tmux new-session -d -s {tmux_project_name} && tmux send-keys -t {tmux_project_name} "{cmd}" Enter'
    exec_remote_cmd(remote_ip, tmux_session_cmd)
    time.sleep(5)

    # 在此处等待进程完成
    while pull_file(TASK_INFO_DIR, remote_addr, os.path.join(remote_path, train_log_name), passwd, port):
        time.sleep(20)

    return True



def model_deploy(remote_ip, project_name):
    remote_path = train_machine_tool_dir[remote_ip]
    remote_model_rel_path = os.path.join("runs", "train", project_name, "weights")
    remote_pt_model_rel_path = os.path.join(remote_model_rel_path, "best.pt")
    remote_pt_model_abs_path = os.path.join(remote_path, remote_pt_model_rel_path)

    # check remote best.pt
    ret = check_remote_file_exist(remote_ip, remote_pt_model_abs_path)
    if not ret:
        SEND_LOG_MSG.error("remote best weights not exists, train could be stop: [%s]"%remote_pt_model_abs_path)
        return False

    # first, ready dir
    # local
    local_model_path = os.path.join(MODEL_REPO_DIR, project_name)
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path)

    remote_onnx_model_rel_path = os.path.join(remote_model_rel_path, "best_op9.onnx")

    remote_result_abs_path = os.path.join(remote_path, remote_model_rel_path, "..", "results.txt")
    remote_onnx_model_abs_path = os.path.join(remote_path, remote_onnx_model_rel_path)

    SEND_LOG_MSG.info("ready to export onnx")
    # export onnx
    cmd = "cd %s && bash -i %s %s"%(remote_path, "./usertools/export_onnx.sh", remote_pt_model_rel_path)
    exec_remote_cmd(remote_ip, cmd)

    remote_addr, passwd, port = train_machine_info[remote_ip]

    SEND_LOG_MSG.info("ready to copy train content into local")
    # copy to local
    ret = pull_file(local_model_path, remote_addr, remote_result_abs_path, passwd, port)
    if not ret:
        SEND_LOG_MSG.error("Failed to pull result: [%s]"%remote_result_abs_path)
        return False
    ret = pull_file(local_model_path, remote_addr, remote_onnx_model_abs_path, passwd, port)
    if not ret:
        SEND_LOG_MSG.error("Failed to pull onnx: [%s]"%remote_onnx_model_abs_path)
        return False
    ret = pull_file(local_model_path, remote_addr, remote_pt_model_abs_path, passwd, port)
    if not ret:
        SEND_LOG_MSG.error("Failed to pull pt model: [%s]"%remote_pt_model_abs_path)
        return False

    candidate_list = [
        ["xavier", "trt8"],
        ["1080ti", "trt8"],
        ["tx2", "trt4"],
        ["1080ti", "trt4"],
    ]

    SEND_LOG_MSG.info("ready to copy cvt onnx model")
    # cvt to engine
    ret_local_path = engine_cvt_pipeline("best_op9.onnx", candidate_list, local_model_path)
    SEND_LOG_MSG.info(ret_local_path)
    SEND_LOG_MSG.info("all done!")
    return True

def text_to_html(text):
    # 替换空格和制表符为HTML实体
    text = text.replace(' ', '&nbsp;').replace('\t', '&emsp;')

    # 替换换行符为HTML换行标签
    text = text.replace('\n', '<br>')

    # 使用<pre>标签来保留文本的格式
    html_text = f'<pre>{text}</pre>'

    return html_text

def send_email(email, project_name):
    local_model_path = os.path.join(MODEL_REPO_DIR, project_name)
    subject = "Model Train System: Please Check Your Model Train Result by [%s]"%project_name

    email_content = '''
Dear %s,
    I've finished train task for you, and guys, that was quite a task (mischievous).
    You can download it from this website: http://%s:22222/

        user email: %s;
        project   : %s;

    Wishing you a day full of magic and wonder!
Love,
Cheng
                '''%(
                    email.split('@')[0],
                    HOST_IP,
                    email,
                    project_name
                )

    # attachments_list = []

    # for a in os.listdir(TASK_INFO_DIR):
    #     attachments_list.append(os.path.join(TASK_INFO_DIR, a))

    send_flag = send_mail_personal('lpci@uisee.com',
        [email],
        text_to_html(email_content),
        ["task_info/remote_train.out", "task_info/%s.log"%project_name],
        subject
    )

    SEND_LOG_MSG.info("Send Email Done!")
    return send_flag

def main(email, remote_ip, project_name, base_model, start_progress = -1):
    if os.path.exists(PROC_DIR) and start_progress == -1:
        SEND_LOG_MSG.error("remote train has been execed!")
        return

    if start_progress <= 0:
        train_task_progress = {}
        train_task_progress["status"] = True
        train_task_progress["task_progress"] = 0
        serialize_data(train_task_progress, PROC_DIR)
    else:
        train_task_progress=deserialize_data(PROC_DIR)
        train_task_progress["status"] = True

    # 数据同步
    if start_progress <= 1:
        train_task_progress["task_progress"] = 1
        SEND_LOG_MSG.info("ready to data sync.")
        serialize_data(train_task_progress, PROC_DIR)
        ret = data_sync(remote_ip)
        train_task_progress["status"] = ret
        if not ret:
            serialize_data(train_task_progress, PROC_DIR)
            return False

    # 开始训练
    if start_progress <= 2:
        train_task_progress["task_progress"] = 2
        SEND_LOG_MSG.info("ready to launch train")
        serialize_data(train_task_progress, PROC_DIR)
        ret = launch_train(remote_ip, base_model, project_name)
        train_task_progress["status"] = ret
        if not ret:
            serialize_data(train_task_progress, PROC_DIR)
            return False

    #量化部署
    if start_progress <= 3:
        train_task_progress["task_progress"] = 3
        SEND_LOG_MSG.info("ready to deploy")
        serialize_data(train_task_progress, PROC_DIR)
        ret = model_deploy(remote_ip, project_name)
        train_task_progress["status"] = ret
        if not ret:
            serialize_data(train_task_progress, PROC_DIR)
            return False

    #发送邮件
    if start_progress <= 4:
        train_task_progress["task_progress"] = 4
        SEND_LOG_MSG.info("ready to send email")
        serialize_data(train_task_progress, PROC_DIR)
        ret = send_email(email, project_name)
        train_task_progress["status"] = ret
        if not ret:
            serialize_data(train_task_progress, PROC_DIR)
            return False

    train_task_progress["task_progress"] = -1
    serialize_data(train_task_progress, PROC_DIR)
    SEND_LOG_MSG.info("pipe done")

    return True


if __name__=="__main__":

    import argparse
    import setproctitle

    setproctitle.setproctitle("light_remote_train")

    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default="xiaoqiang.cheng@uisee.com")
    parser.add_argument('--remote-ip', type=str, default="10.0.93.231")
    parser.add_argument('--project', type=str, default="uisee")
    parser.add_argument('--base-model', type=str, default="")
    parser.add_argument('--start-progress', type=int, default=-1)

    args = parser.parse_args()

    main(args.email, args.remote_ip, args.project, args.base_model, args.start_progress)

    # TODO: 在脚本正常退出时 杀死远程进程
