
import streamlit as st
from config import *
from utils import *
import time
from remote_train import get_remote_gpus_info
import os
import pandas as pd
from streamlit_react_flow import react_flow
import streamlit_authenticator as stauth

st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="📡",
            )

# 用户信息，后续可以来自DB
#names = ['程晓强', '管理员'] # 用户名
#usernames = ['cxq10490', 'admin']  # 登录名
#passwords = ['888888', '888888']  #登录密码
# 对密码进行加密操作，后续将这个存放在credentials中
#hashed_passwords = stauth.Hasher(passwords).generate()

# 定义字典，初始化字典
#credentials = {'usernames': {}}
# 生成服务器端的用户身份凭证信息
#for i in range(0, len(names)):
#    credentials['usernames'][usernames[i]] = {'name': names[i], 'password': hashed_passwords[i]}
#authenticator = stauth.Authenticate(credentials, 'some_cookie_name', 'some_signature_key', cookie_expiry_days=1)
#name, authentication_status, username = authenticator.login('Login', 'main')

authenticator = 0
st.session_state.task_progress = -1
st.session_state.task_status = False
st.session_state.run_status = True
st.session_state.last_train_cfg = []
st.session_state.machine_gpus_info = {}

if "rerun" not in st.session_state.keys():
    st.session_state["rerun"] = True
    history_config = deserialize_data("database.pkl")

    print("=============================================")

    if history_config is None:
        pass
    else:
        for key, value in history_config.items():
            st.session_state['key'] = value


if os.path.exists(PROC_DIR):
    train_task_progress = deserialize_data(PROC_DIR)
    st.session_state.task_progress = train_task_progress['task_progress']
    if "status" in train_task_progress.keys():
        st.session_state.run_status = train_task_progress['status']
    else:
        st.session_state.run_status = False
    st.session_state.task_status = True



# st_autorefresh(interval=5000, key="axaakjbsdfbipjsdfasbdhj")


def creat_train_pipeline(nodes = [], progress = 0, run_status = True):
    elements = []
    if run_status:
        special_color = "#ffc500"
    else:
        special_color = "#ff0000"

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
        first, second, third = st.sidebar.columns([1,2,2], gap='small')
        with first:
            status = st.checkbox("G%d:"%key)
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
    default_value.sort()
    default_value = ["scratch"] + default_value[::-1]
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
    machine_list = list(train_machine_info.keys())
    if not os.path.exists(PROC_DIR) or "target_train_machine" not in st.session_state.keys():
        default_machine_index = 0
    else:
        default_machine_index = machine_list.index(st.session_state.target_train_machine)
    target_train_machine = st.sidebar.selectbox("选择服务器", machine_list, index=default_machine_index, label_visibility="collapsed")

    if "target_train_machine" not in st.session_state.keys() or target_train_machine != st.session_state.target_train_machine or st.session_state["rerun"]:
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
    target_train_batch_size = create_input_txt_with_label("batch size/gpu", "16")
    target_train_base_model = create_selectbox_with_label("base model", os.listdir(MODEL_REPO_DIR))

    st.session_state.target_train_gpu = ",".join(map(str, target_train_gpu))
    print(target_train_gpu, st.session_state.target_train_gpu)
    st.session_state.target_train_project_name = target_train_project_name
    st.session_state.target_train_epoch = target_train_epoch
    st.session_state.target_train_worker_num = target_train_worker_num
    st.session_state.target_train_batch_size = str(int(target_train_batch_size) * len(target_train_gpu))
    if target_train_base_model == "scratch":
        st.session_state.target_train_base_model = ''
        # st.session_state.target_train_epoch = "300"
        st.session_state.target_train_hyp = "data/hyp.scratch.tiny.yaml"
    else:
        st.session_state.target_train_base_model = target_train_base_model
        st.session_state.target_train_hyp = "data/hyp.finetune.yaml"

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
    script_name = task_name + ".py"
    target_args = [script_name] + args
    print(target_args)
    target_args_str = " ".join(target_args)
    cmd = "nohup python %s > "%target_args_str + "%s/%s.out 2>&1 & "%(TASK_INFO_DIR, task_name)
    SEND_LOG_MSG.info(cmd)
    os.system(cmd)


def stop_train_task():
    if not st.session_state.task_status: return
    # need stop task
    st.session_state.task_status = False
    st.session_state.task_progress = -1

    SEND_LOG_MSG.info("clear cache and stop train task.")
    for i in range(5):
        # kill local process remote process
        os.system("killall ssh")
        os.system("killall light_remote_train")
        os.system("killall sshpass")
        os.system("rm -r %s"%PROC_DIR)
        time.sleep(0.01)

def start_train_task():
    if st.session_state.task_status: return
    st.session_state.task_status = True
    if not os.path.exists(PROC_DIR):
        # need start task, write
        with open("template/launch_train.sh", "w") as f:
            last_launch_script = launch_train_template
            selected_dataset_str = " ".join(st.session_state.selected_dataset)

            last_launch_script = last_launch_script.replace("$dataset_list", selected_dataset_str)
            last_launch_script = last_launch_script.replace("$worker_num", st.session_state.target_train_worker_num)
            last_launch_script = last_launch_script.replace("$gpu_num", str(len(st.session_state.target_train_gpu.split(','))))
            last_launch_script = last_launch_script.replace("$device_num", st.session_state.target_train_gpu)
            last_launch_script = last_launch_script.replace("$batch_size", st.session_state.target_train_batch_size)
            last_launch_script = last_launch_script.replace("$epoch_num", st.session_state.target_train_epoch)
            last_launch_script = last_launch_script.replace("$project_name", st.session_state.target_train_project_name)

            if st.session_state.target_train_base_model == '':
                last_launch_script = last_launch_script.replace("$base_model", '\'\'')
            else:
                last_launch_script = last_launch_script.replace("$base_model", os.path.join(MODEL_REPO_DIR, st.session_state.target_train_base_model + ".pt"))

            last_launch_script = last_launch_script.replace("$train_hyp", st.session_state.target_train_hyp)
            SEND_LOG_MSG.info(last_launch_script)
            f.write(last_launch_script)

            st.session_state.task_progress = 0
            if st.session_state.target_train_base_model != "":
                train_cfg = ['--email', st.session_state.user_email,
                    '--remote-ip', st.session_state.target_train_machine,
                    '--project', st.session_state.target_train_project_name,
                    '--base-model', st.session_state.target_train_base_model]
            else:
                train_cfg = ['--email', st.session_state.user_email,
                        '--remote-ip', st.session_state.target_train_machine,
                        '--project', st.session_state.target_train_project_name
                    ]

            st.session_state.last_train_cfg = train_cfg
            serialize_data(train_cfg, "train_cfg.pkl")
            exec_script_task("remote_train", train_cfg)
    else:
        SEND_LOG_MSG.warning("have launch train, do not click again!")

def save_and_utgz_uploaded_file(uploadedfile):
    tgz_name = os.path.join(UPLOAD_TMP_DIR, uploadedfile.name)
    with open(tgz_name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    os.system("tar -xf %s -C %s"%(tgz_name, DATASET_DIR))
    return st.success("upload successful, save to %s"%DATASET_DIR)

def update_dataset_table():
    folders = os.listdir(DATASET_DIR)
    for f in folders:
        if "dataset" in f:
            try:
                with open(os.path.join(DATASET_DIR, f, "tag"), "r") as fi:
                    tag = fi.read().replace("\n", "")
            except:
                tag = "CAN NOT FIND"

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

def main_ui_layout():
    first, second = st.columns([1,2], gap='small')
    if "dataset" not in st.session_state.keys():
        st.session_state.dataset = {}
        update_dataset_table()

    with first:
        st.markdown("### 数据集管理")
        st.markdown("-----")
        st.markdown("#### 上传新数据")
        upload_file = st.file_uploader("上传新数据", "tar.gz", label_visibility = "collapsed")
        if upload_file is not None:
            save_and_utgz_uploaded_file(upload_file)
            update_dataset_table()
        st.markdown("-----")
        data_df, scene_num, train_sum, val_sum = dataset_to_pd_frame(st.session_state.dataset)
        st.markdown("**scene_num:** %d\t **train_num:** %d\t **val_sum:** %d"%(scene_num, train_sum, val_sum))
        dataset_table_list = st.data_editor(
            data_df,
            column_config={
                "enable": st.column_config.CheckboxColumn(
                    "enable",
                    default=False,
                )
            },
            width = 400,
            height = 1500,
            disabled=["dataset", "tag", "train", "val"],
            hide_index=False,
        )

        mask = dataset_table_list.enable == True
        st.session_state.selected_dataset =list(dataset_table_list.dataset[mask])

    with second:
        st.markdown("### 模型训练")
        st.markdown("------")
        col1, col2 = st.columns([1, 1], gap='small')
        with col1:
            with st.expander("训练配置", expanded=True):
                st.session_state.user_email = st.selectbox("**负责人邮箱**：(用于接收训练结果信息)",
                        owners_email, label_visibility="collapsed")
                if not isValid(st.session_state.user_email):
                    st.info("请输入有效的邮箱地址！")
                    return

                start_train_button = st.checkbox("开始训练", value=st.session_state.task_status)
                if start_train_button:
                    st.info("已经执行开始训练，请勿重复点击...")
                    start_train_task()
                else:
                    st.info("休息中...")
                    stop_train_task()

                st.markdown("[%s](http://%s)"%("查看TensorBoard",
                    st.session_state.target_train_machine + ":6001"),
                                unsafe_allow_html=True)

                st.markdown("[%s](http://%s)"%("查看模型管理页面",
                    (HOST_IP + ":22222/download")),
                                unsafe_allow_html=True)


                if len(st.session_state.selected_dataset) == 0:
                    selected_dataset_str = "all"
                else:
                    selected_dataset_str = len(st.session_state.selected_dataset)
                train_summary_info = '''
                    ##### light train system summary: \n
                        email  : %s;\n
                        Project: %s;\n
                        Server : %s;\n
                        GPUS   : %s;\n
                        Batch  : %s;\n
                        Epoch  : %s;\n
                        Dataset: %s;\n
                '''%(
                    st.session_state.user_email,
                    st.session_state.target_train_project_name,
                    st.session_state.target_train_machine,
                    st.session_state.target_train_gpu,
                    st.session_state.target_train_batch_size,
                    st.session_state.target_train_epoch,
                    selected_dataset_str
                )
                st.info(train_summary_info)

        with col2:
            with st.expander("查看训练进度", expanded=True):
                #creat_train_pipeline(["启动流程", "数据同步", "更新模型", "量化部署", "发送邮件", "结束流程"],
                #                st.session_state.task_progress, run_status = st.session_state.run_status)

                continue_button = st.button("任务中断，点击继续", disabled=st.session_state.run_status, use_container_width=True)
                if continue_button:
                    train_cfg = deserialize_data("train_cfg.pkl")
                    print(train_cfg)
                    st.session_state.run_status = True
                    exec_script_task("remote_train",
                        train_cfg + ["--start-progress", str(st.session_state.task_progress)])
                print( st.session_state.task_progress, st.session_state.run_status)
                creat_train_pipeline(["启动流程", "数据同步", "更新模型", "量化部署", "发送邮件", "结束流程"],
                                st.session_state.task_progress, run_status = st.session_state.run_status)


def test():
    pass


def main():
    #if authentication_status:
    sidebar_ui_layout()
    main_ui_layout()
    serialize_data(st.session_state.to_dict(), "database.pkl")
    #elif authentication_status == None:
    #    st.info("Please input username and password")
    #else:
    #    st.error('Username/password is incorrect')


main()


