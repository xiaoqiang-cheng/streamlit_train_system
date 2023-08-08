
import streamlit as st
from config  import *
import os
import numpy as np


import pandas as pd

st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="📡",
            )


model_history_list = os.listdir(MODEL_REPO_DIR)
model_history_list.sort()

def create_download_button(target_path):
    label = os.path.basename(target_path)
    with open(target_path, 'rb') as f:
        ret = st.download_button(label, f, key = target_path, file_name=label)
    return ret


def read_results_list(results_path):
    ret = []
    with open(results_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        ret.append(line.split())

    return ret


for folder in model_history_list:
    st.markdown("------")

    curr_results_dir = os.path.join(MODEL_REPO_DIR, folder)
    curr_results_dir_list = os.listdir(curr_results_dir)
    curr_results_dir_list.sort()
    result_path = os.path.join(curr_results_dir, "results.txt")
    result_table = read_results_list(result_path)
    mAP_curve = np.array(result_table)[:, 10]
    mAP_curve = list(map(float, mAP_curve))
    info_summary = '''
        project: %s ======> mAP: %.4f;
    '''%(folder, mAP_curve[-1])
    st.info(info_summary)

    chart_data = pd.DataFrame(
        mAP_curve,  columns=["mAP"])
    st.line_chart(chart_data, height = 200)

    col_layouts = st.columns(6)

    for index, engine in enumerate(curr_results_dir_list):
        engine_dir = os.path.join(curr_results_dir, engine)
        with col_layouts[index]:
            create_download_button(engine_dir)

