
import streamlit as st


st.set_page_config(layout="wide",
            page_title="Light Train System",
            page_icon="📡",
            )

with st.spinner():
    taskinfo = st.code("start train task:")
    with open("task_info/remote_train.out", "r") as f:
        console_str = f.read()
    taskinfo.text(console_str)