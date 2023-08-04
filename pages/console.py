
import streamlit as st
from streamlit_autorefresh import st_autorefresh


with st.spinner():
    taskinfo = st.code("start train task:")
    with open("log/model_deploy.log", "r") as f:
        console_str = f.read()
    taskinfo.text(console_str)