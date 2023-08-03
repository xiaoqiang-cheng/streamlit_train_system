
import streamlit as st
from config import *
from utils import *
import time
from remote_train import get_remote_gpus_info
import os
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from streamlit_react_flow import react_flow

st.session_state.task_progress = -1


if not os.path.exists("database.pkl"):
    st.session_state.task_status = False
else:
    history_config = deserialize_data("database.pkl")
    st.session_state.task_status = True

    if history_config is None:
        st.session_state.task_status = False
    else:
        for key, value in history_config.items():
            st.session_state[key] = value

st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="🤖",
            )



# def creat_control_flow_unit(label, callback_fun):
#     button =  st.button(label, key=label, on_click=callback_fun)
#     status_bar = st.progress(0)
#     return status_bar

def creat_control_flow_unit(curr_index = 0):
    with st.expander("See explanation"):
        st.button("启动训练")

def creat_train_pipeline(nodes = [], progress = 0):
    elements = []
    special_color = "#ffc500"

    for i, n in enumerate(nodes):
        node = {
            "id" : str(i),
            "data" : {
                "label": n
            },
            "style": {
                "width": 200
            },
            "position": { "x": 100 , "y": 100* (i + 1) }
        }

        if (progress == i):
            node["style"]["background"] = special_color
        elements.append(node)

    edges = []
    for i, n in enumerate(nodes[:-1]):
        edge = {
            "id": 'e%d-%d'%(i, i+1),
            "source": str(i),
            "target": str(i + 1),
            "animated": bool(i == progress)
        }
        edges.append(edge)


    flowStyles = { "height": 1000, "width":500 }

    elements.extend(edges)
    react_flow("pipeline", elements=elements, flow_styles=flowStyles)



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

def create_input_txt_with_label_in_main(label, default_value):

    st.markdown(label)
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

    st.session_state.target_train_gpu = ",".join(map(str, target_train_gpu))
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

def exec_script_task(task_name, args = []):
    script_name = os.path.join("script", task_name + "py")
    target_args = [script_name] + args
    target_args_str = " ".join(target_args)
    cmd = "nohup python %s > "%target_args_str + "%s/%s.out 2>&1 & "%(TASK_INFO_DIR, task_name)
    SEND_LOG_MSG.info(cmd)
    os.system(cmd)

def submit_train_task():
    if st.session_state.task_status:
        # need stop task
        st.session_state.task_status = False
        st.info("正在停止训练，请勿重复点击...")
        os.system("rm -r database.pkl")
        SEND_LOG_MSG.info("clear cache and stop train task.")
    else:
        serialize_data(st.session_state.to_dict(), "database.pkl")
        import copy
        session_state = copy.deepcopy(st.session_state.to_dict())
        st.session_state.task_status = True
        st.info("正在启动训练，请勿重复点击...")
        # need start task, write
        with open("template/launch_train.sh", "w") as f:
            last_launch_script = launch_train_template
            last_launch_script = last_launch_script.replace("$worker_num", st.session_state.target_train_worker_num)
            last_launch_script = last_launch_script.replace("$device_num", st.session_state.target_train_gpu)
            last_launch_script = last_launch_script.replace("$batch_size", st.session_state.target_train_batch_size)
            last_launch_script = last_launch_script.replace("$epoch_num", st.session_state.target_train_epoch)
            last_launch_script = last_launch_script.replace("$project_name", st.session_state.target_train_epoch)
            last_launch_script = last_launch_script.replace("$base_model", st.session_state.target_train_base_model)

            SEND_LOG_MSG.info(last_launch_script)
            f.write(last_launch_script)

        print("==============================", session_state['task_progress'])
        st.write("start task")
        session_state['task_progress'] = 0
        serialize_data(session_state, "database.pkl")
        # 首先数据同步

        time.sleep(10)

        print("==============================", session_state['task_progress'])
        st.write("start data sync")
        session_state['task_progress'] = 1
        serialize_data(st.session_state.to_dict(), "database.pkl")
        # 开始训练
        print("==============================", session_state['task_progress'])
        time.sleep(10)
        st.write("start train")
        session_state['task_progress']  = 2
        serialize_data(st.session_state.to_dict(), "database.pkl")
        # 开始量化部署
        print("==============================", session_state['task_progress'])
        time.sleep(10)
        st.write("start deploy")
        session_state['task_progress']  = 3

        print("==============================", session_state['task_progress'])
        time.sleep(10)
        st.write("start send")
        session_state['task_progress'] = 4

        time.sleep(10)

        print("==============================", session_state['task_progress'])
        session_state['task_progress'] = -1
        serialize_data(st.session_state.to_dict(), "database.pkl")
        session_state['task_status']  = False

        st.info("上一次训练已结束")
        os.system("rm -r database.pkl")


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

    with second:
        st.markdown("### 模型训练")
        st.markdown("------")
        col1, col2 = st.columns([1, 1], gap='small')
        with col1:
            with st.expander("训练配置", expanded=True):
                create_input_txt_with_label_in_main("**负责人邮箱**：(用于接收训练结果信息)", "xiaoqiang.cheng@uisee.com")
                if st.session_state.task_status:
                    task_button_label = "停止训练"
                else:
                    task_button_label = "开始训练"
                start_train_button = st.button(task_button_label, use_container_width=True)
                if start_train_button:
                    submit_train_task()


                st.markdown("[%s](http://%s)"%("查看TensorBoard",
                    st.session_state.target_train_machine + ":6001"),
                                unsafe_allow_html=True)

                train_summary_info = '''
                    ##### light train system summary: \n
                        email  : \n
                        server : \n
                        GPUS   : \n
                        Project: \n
                        Batch  : \n
                        Epoch  : \n
                '''
                st.info(train_summary_info)

        # if os.path.exists("database.pkl"):
        #     history_config = deserialize_data("database.pkl")
        #     for key, value in history_config.items():
        #         st.session_state[key] = value

        with col2:
            with st.expander("查看训练进度", expanded=True):
                creat_train_pipeline(["启动训练", "数据同步", "更新模型", "量化部署", "发送邮件", "结束训练"],
                                        st.session_state.task_progress)

def test():
    pass


def main():
    sidebar_ui_layout()
    main_ui_layout()



main()

if st.session_state.task_status:
    serialize_data(st.session_state.to_dict(), "database.pkl")

