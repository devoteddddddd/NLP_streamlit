import torch

import streamlit as st
st.title('My first app')

st.write('After split', torch.cuda.is_available())


