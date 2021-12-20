import streamlit as st 
import pandas as pa
from sklearn import datasets

st.write("""
# Iris Classifier    
Aplikasi ini digunakan untuk mengklasifikasikan bunga Iris berdasarkan ukuran Sepal dan Petal. 
""") 

st.sidebar.header("Masukan Ukuran Sepal dan Petal")

def iris_parameters():
    sl = st.sidebar.slider('Panjang Sepal', 4.0, 8.0, 5.0)
    sw = st.sidebar.slider('Lebar Sepal', 1.0, 5.0, 3.0)
    pl = st.sidebar.slider('Panjang Petal', 1.0, 7.0, 2.0)
    pw = st.sidebar.slider('Lebar Petal', 0, 4, 1) 

    data = {'sl': sl, 
            'sw': sw,
            'pl': pl, 
            'pw': pw}
    parameters = pa.DataFrame(data, index=[0])
    return parameters

dp = iris_parameters()
st.subheader('Ukuran Sepal dan Petal')
st.write(dp) 
