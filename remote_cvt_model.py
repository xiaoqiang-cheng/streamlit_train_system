import sys
import subprocess
import os
from config import *
import multiprocessing
from send import send_mail_personal


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

def exec_remote_cmd(ctype, cmd):
    remote_addr, pwd, port = machine_info[ctype]
    command = "sshpass -p %s ssh -p%s %s \"%s\""%(pwd, port, remote_addr, cmd)
    SEND_LOG_MSG.info(command)
    process = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return True

def exec_onnx2trt(onnx_name, machine_type, etype = TRT4_INDEX):
    SEND_LOG_MSG.info("start [%s] cvt [%s] [%s] process"%(onnx_name, machine_type, engine_type_str[etype]))
    tool_path = machine_tool_dir[machine_type][etype]
    onnx_path = os.path.join(MODEL_DEPLOY_SAVE_DIR, onnx_name)
    trt_path = os.path.join(MODEL_DEPLOY_SAVE_DIR, onnx_name + ".bin")
    remote_addr, pwd, port = machine_info[machine_type]

    remote_onnx_path = os.path.join(tool_path, onnx_path)
    remote_trt_path = os.path.join(tool_path, trt_path)
    local_trt_path = os.path.join(MODEL_DEPLOY_SAVE_DIR,
                onnx_name + ".%s"%machine_type + ".%s"%engine_type_str[etype] + ".bin")

    # first, push model to remote machine
    status = push_file(onnx_path, remote_addr, remote_onnx_path, pwd, port)
    if not status:
        SEND_LOG_MSG.error("push file error, please check network or onnx file")
        return None
    # second, start to cvt model
    cmd = "cd %s && ./onnx2trt.sh %s"%(tool_path, onnx_path)
    status = exec_remote_cmd(machine_type, cmd)
    if not status:
        SEND_LOG_MSG.error("cvt model error, please check remote machine")
        return None
    # lastly, pull trt file
    status = pull_file(local_trt_path, remote_addr, remote_trt_path, pwd, port)
    if not status:
        SEND_LOG_MSG.error("pull file error, please check remote machine")
        return None
    SEND_LOG_MSG.info("model cvt successful! addr: %s"%local_trt_path)
    return local_trt_path



def exec_thread(params):
    return exec_onnx2trt(*params)



def engine_cvt_pipeline(onnx_name, candidate_list):
    need_cvt_list = []
    for c in candidate_list:
        need_cvt_list.append([onnx_name, c[0], engine_type_str.index(c[1])])
        SEND_LOG_MSG.info("cvt model detail: %s"%(str(need_cvt_list[-1])))

    p = multiprocessing.Pool(len(need_cvt_list))
    ret = p.map(exec_thread, need_cvt_list)
    p.close()
    p.join()
    return ret
    # for c in candidate_list:
    #     ret = exec_onnx2trt(onnx_name, *c)

def text_to_html(text):
    # 替换空格和制表符为HTML实体
    text = text.replace(' ', '&nbsp;').replace('\t', '&emsp;')

    # 替换换行符为HTML换行标签
    text = text.replace('\n', '<br>')

    # 使用<pre>标签来保留文本的格式
    html_text = f'<pre>{text}</pre>'

    return html_text


if __name__ == "__main__":
    import numpy as np

    # candidate_list = [
    #     ["xavier", "trt8"],
    #     ["tx2", "trt4"],
    #     ["1080ti", "trt4"],
    #     ["1080ti", "trt8"]
    # ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, default="xiaoqiang.cheng@uisee.com")
    parser.add_argument('--onnx', type=str, default="test.onnx")
    parser.add_argument('--clist', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    print(args)
    # print(args.clist.reshape(-1, 2))
    cand_lists = np.reshape(args.clist, (-1, 2))

    print(cand_lists)
    if len(args.clist) == 0:
        pass
    else:
        ret = engine_cvt_pipeline(args.onnx, cand_lists)
        print(ret)
        subject = "Model CVT: Please Check Your Model Engine by [%s]"%args.onnx

        email_content = '''
Dear %s,
    I've finished transferring all the models for you, and guys, that was quite a task (mischievous).
    You'll find them attached to this email.

        user email: %s;
        onnx file : %s;
        target engine : \n\t\t\t\t\t\t%s;

    Wishing you a day full of magic and wonder!
Love,
Cheng
        '''%(
            args.email.split('@')[0],
            args.email,
            args.onnx,
            "\n\t\t\t\t\t\t".join(map(str, cand_lists))
        )

        send_flag = send_mail_personal('lpci@uisee.com',
            [args.email],
            text_to_html(email_content),
            ret + ["task_info/cvt_console_log.out"],
            subject
        )

        if send_flag:
            print(email_content)
        else:
            back_content = '''
Dear %s,
    I've finished transferring all the models for you, and guys, that was quite a task (mischievous).
    The attachment may have failed to be sent.
    You can download it from this website: http://10.0.89.150:9090/

        user email: %s;
        onnx file : %s;
        target engine : \n\t\t\t\t\t\t%s;

    Wishing you a day full of magic and wonder!
Love,
Cheng
            '''%(
                    args.email.split('@')[0],
                    args.email,
                    args.onnx,
                    "\n\t\t\t\t\t\t".join(map(str, cand_lists))
                )
            send_mail_personal('lpci@uisee.com',
                [args.email],
                text_to_html(back_content),
                ["task_info/cvt_console_log.out"],
                subject
            )
            print(back_content)
        print("pipeline done!")

