import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd



st.title('My first app')

content = st.text_input('Sentence')

l = []
for i in content:
    l.append(i)

st.write('After split', l)

