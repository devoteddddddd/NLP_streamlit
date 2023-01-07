
import streamlit as st
import time
import jieba
from gensim.models import Word2Vec
import torch
import torch.nn as nn

st.set_page_config(page_title="首页")

st.title('APP简介')
st.write("我们做的是一个简单的文本分类的Web应用，我们将文本分类分为2个子模块：主题分类和情感分类....")
st.write("作者：全晨、于永孜、徐黛涵")
st.sidebar.success("请选择一个功能")

