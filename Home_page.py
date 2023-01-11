
import streamlit as st
import time
import jieba
from gensim.models import Word2Vec
import torch
import torch.nn as nn

st.set_page_config(page_title="首页")


st.write("# APP简介 👋")

st.markdown(
        """
        我们小组实现了一个中文文本分类的web应用。我们将中文文本分类分为2个子模块：情感分类和新闻主题分类。这2个子模块互相独立。\n
        对于情感分类，部署在这里的是针对电影评论进行二分类：积极和消极。\n
        对于新闻主题分类，是根据新闻的标题预测新闻的主题。我们实现的是14分类，
        分别是： 财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。


        ### 使用注意事项

        - 一次只能预测一个电影评论或新闻标题。
        - 不能什么都不输入就直接点"运行"，会出现错误。
        - 预测完一个电影评论或新闻标题之后，全选文本框所有文字手动删除，再输入新的。
        - 我们提供了电影评论和新闻标题的测试用例供用户输入，存在txt文件里，用户可以随意将里面的一些文本用例复制过来进行预测，具体见PDF报告中附录那块~。
        - 第一次预测时会稍微慢一点点，因为要加载模型，但第二次及以后会快很多，因为我们使用了cache机制。
        - 本网页APP兼容电脑端和手机端。
        
        ### 关于作者

        - 全晨，        Email: 3182815986@qq.com
        - 于永孜，      Email: yuyz@bit.edu.cn
        - 徐黛涵，      Email: daihan-xu@qq.com
    """
)


st.sidebar.success("请选择一个功能")

