
import streamlit as st
import time
import jieba
from gensim.models import Word2Vec
import torch
import torch.nn as nn

st.set_page_config(page_title="首页")

#st.title('APP简介')
#st.write("我们做的是一个简单的文本分类的Web应用，我们将文本分类分为2个子模块：主题分类和情感分类....")
#st.write("作者：全晨、于永孜、徐黛涵")

st.write("# Welcome to Streamlit! 👋")

st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
)


st.sidebar.success("请选择一个功能")

