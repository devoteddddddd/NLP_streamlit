import torch

import streamlit as st
st.title('My first app')

a = torch.cuda.is_available ()
st.write('After split', a)


