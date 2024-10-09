
import streamlit as st
from config  import *
import os
import numpy as np
import re
import functools
import pandas as pd

st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="📡",
            )

def model_compare(x, y):
    x_tail_time = x[-16:]
    y_tail_time = y[-16:]

    xt = int("".join(re.findall(r'\d+', x_tail_time)))
    yt = int("".join(re.findall(r'\d+', y_tail_time)))
    if xt > yt:
        return -1
    else:
        return 1

model_history_list = os.listdir(MODEL_REPO_DIR)
model_history_list.sort(key=functools.cmp_to_key(model_compare))

# 使用缓存机制来读取文件
@st.cache_data
def get_file_content(file_path):
    with open(file_path, "rb") as file:
        return file.read()

def create_download_button(target_path):
    label = os.path.basename(target_path)
    # with open(target_path, 'rb') as f:
    ret = st.download_button(label,
            data=get_file_content(target_path),
            key = target_path,
            file_name=label,
            mime="application/octet-stream")
    return ret


def read_results_list(results_path):
    ret = []
    with open(results_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        ret.append(line.split())

    return ret


for folder in model_history_list:
    curr_results_dir = os.path.join(MODEL_REPO_DIR, folder)
    curr_results_dir_list = os.listdir(curr_results_dir)
    curr_results_dir_list.sort()
    result_path = os.path.join(curr_results_dir, "results.txt")
    result_table = read_results_list(result_path)
    mAP_curve = np.array(result_table)[:, 10]
    mAP_curve = list(map(float, mAP_curve))
    info_summary = '''
%s \t\t======> mAP: %.4f;
    '''%(folder, mAP_curve[-1])

    with st.expander(info_summary):
        chart_data = pd.DataFrame(
            mAP_curve,  columns=["mAP"])
        st.line_chart(chart_data, height = 200)

        col_layouts = st.columns(len(curr_results_dir_list))

        for index, engine in enumerate(curr_results_dir_list):
            engine_dir = os.path.join(curr_results_dir, engine)
            with col_layouts[index]:
                create_download_button(engine_dir)

