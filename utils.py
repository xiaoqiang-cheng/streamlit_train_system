from log_sys import XQLOGHandler
import re

SEND_LOG_MSG = XQLOGHandler()


regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')

def isValid(email):
    if re.fullmatch(regex, email):
        return True
    else:
        return False


def exec_cvt_task(email, onnx_name, clist):
    import numpy as np
    import os
    cand_lists = np.reshape(clist, (-1))
    list_str = " ".join(cand_lists)
    cmd = "nohup python remote_cvt_model.py --email %s --onnx %s --clist %s > %s/cvt_console_log.out 2>&1 & "%(email, onnx_name, list_str, TASK_INFO_DIR)
    os.system(cmd)
    print(cmd)
    return cmd

def gen_html_footer():

    need_show_label = [
        # url, label
        ["http://10.0.89.150:9999/", "红绿灯测试demo"],
        ["", "Author:xiaoqiang.cheng@uisee.com"],
    ]

    label = ""
    for i in need_show_label:
        label = label + "<a href='%s' target='_blank'>%s</a>"%(i[0], i[1]) + " | "

    template = '''
<footer style="border-top: 1px solid ; background-repeat: repeat;" class="footer">
    <div class="footer-inner">
        <div class="footer-copyright" align="center">
            <a> Copyright © 2023-? 模型部署 </a> |
            %s
        </div>
    </div>
</footer>
    '''%label

    return template

import pickle
def serialize_data(data:dict, file_path):
    try:
        data.pop("rerun")
        data.pop("task_status")
    except:
        pass
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
    except Exception as e:
        print("序列化数据时出现错误:", e)

def deserialize_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print("反序列化数据时出现错误:", e)
        return None


def get_datatime_tail():
    from datetime import datetime
    # 获取当前时刻
    current_time = datetime.now()
    # 将时间对象格式化为字符串，精确到分
    formatted_time = current_time.strftime("-%Y-%m-%d_%H_%M")
    return formatted_time
