
import streamlit as st
from config import *
from utils import *
import time
from remote_train import get_remote_gpus_info
import os
import pandas as pd

test_gpu_dict = "{0: {'status': '11.5M/11178.5M', 'percentage': 20}, 1: {'status': '11.5M/11178.5M', 'percentage': 0}, 2: {'status': '704.8M/11161.6M', 'percentage': 0}}"

st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="🤖",
            )


def creat_progress_with_label(gpu_dict:dict):
    ret = []
    for key, val in gpu_dict.items():
        first, second, third = st.sidebar.columns([1,2,1], gap='small')
        with first:
            status = st.checkbox("gpu %d:"%key)
        with second:
            st.progress(int(val["percentage"]))
        with third:
            st.write(val['status'])

        if status:
            ret.append(key)
    if ret == []:
        return list(gpu_dict.keys())
    return ret

def create_input_txt_with_label(label, default_value):
    first, second = st.sidebar.columns([1,2], gap='small')
    with first:
        st.markdown("### "+label)
    with second:
        ret = st.text_input(label, value=default_value, label_visibility="collapsed")
    return ret

def create_selectbox_with_label(label, default_value):
    first, second = st.sidebar.columns([1,2], gap='small')
    with first:
        st.markdown("### "+label)
    with second:
        ret = st.selectbox(label, default_value, label_visibility="collapsed")
    return ret

def tmp_warn_log_msg(msg):
    st.sidebar.info(msg,  icon="ℹ️")
    SEND_LOG_MSG.error(msg)


def sidebar_ui_layout():
    st.sidebar.markdown("# 红绿灯模型在线训练")
    st.sidebar.markdown("------")
    st.sidebar.markdown("## 选择服务器")
    target_train_machine = st.sidebar.selectbox("选择服务器", list(train_machine_info.keys()), label_visibility="collapsed")
    if "target_train_machine" not in st.session_state.keys() or target_train_machine != st.session_state.target_train_machine:
        st.session_state.target_train_machine = target_train_machine
        gpus_info = get_remote_gpus_info(target_train_machine)
        try:
            st.session_state.machine_gpus_info = eval(gpus_info)
        except:
            st.session_state.machine_gpus_info = {}
            tmp_warn_log_msg("get target machine [%s] gpus info ERROR, ret msg: %s"%(target_train_machine, gpus_info))

    target_train_gpu = creat_progress_with_label(st.session_state.machine_gpus_info)
    st.sidebar.markdown("------")
    st.sidebar.markdown("## 修改超参数")
    target_train_project_name = create_input_txt_with_label("project name", "uisee")
    target_train_project_name += get_datatime_tail()
    target_train_epoch = create_input_txt_with_label("epoch", "32")
    target_train_worker_num = create_input_txt_with_label("worker num", "32")
    target_train_batch_size = create_input_txt_with_label("batch size", "32")
    target_train_base_model = create_selectbox_with_label("base model", os.listdir(MODEL_REPO_DIR))

    st.session_state.target_train_project_name = target_train_project_name
    st.session_state.target_train_epoch = target_train_epoch
    st.session_state.target_train_worker_num = target_train_worker_num
    st.session_state.target_train_batch_size = target_train_batch_size
    st.session_state.target_train_base_model = target_train_base_model

def dataset_to_pd_frame(dataset_dict):
    '''
        dataset_00xx:
        {
            "tag": xxx
            "xx" : xx
            ...
        }

    ------------
    |name |tag | |
    '''
    ret = {}
    ret["dataset"] = []
    dataset_list = list(dataset_dict.keys())
    dataset_list.sort()
    ret["enable"]  = [False] * len(dataset_list)
    for key in dataset_list:
        ret["dataset"].append(key)
        val = dataset_dict[key]
        for p, v in val.items():
            if p not in ret.keys():
                ret[p] = []
            ret[p].append(v)
    return pd.DataFrame(ret), len(dataset_list), sum(ret["train"]), sum(ret["val"])


def main_ui_layout():
    first, second = st.columns([1,2], gap='small')
    if "dataset" not in st.session_state.keys():
        st.session_state.dataset = {}
        folders = os.listdir(DATASET_DIR)
        for f in folders:
            if "dataset" in f:
                with open(os.path.join(DATASET_DIR, f, "tag"), "r") as fi:
                    tag = fi.read().replace("\n", "")
                with open(os.path.join(DATASET_DIR, f, "train.txt"), "r") as fi:
                    tmp = fi.readlines()
                    train_num = len(tmp)
                with open(os.path.join(DATASET_DIR, f, "val.txt"), "r") as fi:
                    tmp = fi.readlines()
                    val_num = len(tmp)
                st.session_state.dataset[f] = {}
                st.session_state.dataset[f]["tag"] = tag
                st.session_state.dataset[f]["train"] = train_num
                st.session_state.dataset[f]["val"] = val_num
    with first:
        st.markdown("### 数据集管理")
        st.markdown("-----")
        st.markdown("#### 上传新数据")
        st.file_uploader("上传新数据", "tar.gz", label_visibility = "collapsed")
        st.markdown("-----")
        data_df, scene_num, train_sum, val_sum = dataset_to_pd_frame(st.session_state.dataset)
        st.markdown("**scene_num:** %d\t **train_num:** %d\t **val_sum:** %d"%(scene_num, train_sum, val_sum))
        st.data_editor(
            data_df,
            column_config={
                "enable": st.column_config.CheckboxColumn(
                    "enable",
                    default=False,
                )
            },
            width = 500,
            height = 1500,
            disabled=["dataset", "tag", "train", "val"],
            hide_index=False,
        )





def main():
    sidebar_ui_layout()
    main_ui_layout()


if __name__ == "__main__":

    main()
