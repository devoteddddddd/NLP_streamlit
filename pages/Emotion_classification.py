import streamlit as st
import numpy as np
import pandas as pd
import time



st.set_page_config(page_title="情感分类")


st.title('情感分类')
content2 = st.text_area('请输入文本：', key = 1)

l2 = []


if st.button('运行', key=0):
    for i in range(len(content2)):
        l2.append(content2[i])
    l2.reverse()
    st.write('After split', l2)
