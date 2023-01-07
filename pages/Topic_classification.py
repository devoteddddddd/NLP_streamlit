
import streamlit as st
import jieba
from gensim.models import Word2Vec
import torch
import torch.nn as nn

device = torch.device('cpu')
class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes, num_layers, bidirectional):
        super(LSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.bidirectional:
            h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(device)

        out, (_, _) = self.lstm(x, (h0, c0))
        output = self.fc(out[:, -1, :]).squeeze(0)  # 因为有max_seq_len个时态，所以取最后一个时态即-1层
        return output

to_topic = \
{0:'财经', 1:'彩票', 2:'房产', 3:'股票', 4:'家居',
 5:'教育', 6:'科技', 7:'社会', 8:'时尚', 9:'时政',
 10:'体育', 11:'星座', 12:'游戏', 13:'娱乐'}

st.set_page_config(page_title="主题分类")
st.title('主题分类')
content = st.text_area('请输入新闻标题：(一次只能预测一个新闻标题)', key = 0)


if st.button('运行', key = 1):
    with st.spinner('正在加载模型和推理，请稍等....'):
        model = torch.load('best_model_3.pth', map_location=torch.device('cpu'))
        embed_model = Word2Vec.load('skip_gram')
    l = jieba.lcut(content)
    vectors = []
    for i in l:
        try:
            vec = embed_model.wv[i]
        except:
            vec = embed_model.wv['，'] # 未登录的词统一用逗号来代替
        vec2 = vec.tolist()
        vectors.append(vec2)
    inputs = torch.tensor(vectors).unsqueeze(0)
    outputs = model(inputs)
    st.write('Softmax层输出：', outputs)
    pred = torch.argmax(outputs)
    pred_2 = pred.numpy().tolist()
    topic = to_topic[pred_2]
    st.write('预测的主题是：', topic)
    st.success('模型加载和推理成功！')

