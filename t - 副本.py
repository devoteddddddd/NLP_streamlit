import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
st.title('My first app')
content = st.text_input('Sentence')

l = []
for i in content:
    l.append(i)

device = torch.device('cpu')
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 各个维度的含义是 (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = torch.load('best_model.pth', map_location=torch.device('cpu'))

ix_to_tag = {0:'B_person', 1:'I_person', 2:'B_loc', 3:'I_loc', 4:'B_org', 5:'I_org', 6:'O'}


text_id = \
    [38384, 11697, 9548, 5449, 7353, 31760, 33149, 40999, 7074, 26180, 38167, 38586, 36456, 37647, 17314, 38255, 46897, 25194, 18795, 11754, 27018, 10228, 27523, 47861, 27871, 46897, 21248, 36038, 40976, 27172, 34403, 17314, 43281, 21570, 43261, 1014, 2062, 39847, 45421]

inputs = torch.tensor(text_id)
outputs = model(inputs)
_, pred_ = torch.max(outputs.data, 1)
pred_2_ = pred_.numpy().tolist()
results = [ix_to_tag[i] for i in pred_2_]

st.write('After split', content)
st.write('After split', results)


