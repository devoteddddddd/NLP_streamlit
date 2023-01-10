import streamlit as st
import torch
from transformers import BertTokenizer,BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")
# 数值转文字标签
to_topic = \
{0:'对电影持积极情绪', 1:'对电影持消极情绪'}

st.set_page_config(page_title="情感分类")
st.title('情感分类')
content = st.text_area('请输入一则电影评论：(一次只能预测一个电影评论)', key = 2)


if st.button('运行', key = 3):
    with st.spinner('正在加载模型和推理，请稍等....'):
        
        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, ttl=24*3600, allow_output_mutation=True)
        def load_modell():
            return BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2), torch.load('bert_cla3.ckpt', map_location=torch.device('cpu')), BertTokenizer.from_pretrained('bert-base-chinese')
        
        
        model, model2, tokenizer = load_modell()
        model.load_state_dict(model2)
        #tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    
    inputs = tokenizer(content, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs).logits
    st.write('Softmax层输出：', outputs)
    pred = torch.argmax(outputs)
    pred_2 = pred.numpy().tolist()
    topic = to_topic[pred_2]
    st.write('预测的情感是：', topic)
    st.success('模型加载和推理成功！')
