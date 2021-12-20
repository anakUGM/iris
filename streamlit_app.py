import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Image Resources 
iris_setosa_url = "https://github.com/anakUGM/iris/raw/master/iris-setosa.jpg"
iris_virginica_url = "https://github.com/anakUGM/iris/raw/master/iris-virginica.jpg"
iris_versicolor_url = "https://github.com/anakUGM/iris/raw/master/iris-versicolor.jpg"

st.write("""
# Prediksi Jenis Bunga Iris
Aplikasi ini akan memprediksi jenis bunga iris berdasarkan nilai parameter Sepal dan Petal.
""")

st.sidebar.header('Parameter Sepal dan Petal')

def iris_parameters():
    sepal_length = st.sidebar.slider('Panjang Sepal', 1, 100, 59)
    sepal_width = st.sidebar.slider('Lebar Sepal', 1, 100, 30)
    petal_length = st.sidebar.slider('Panjang Petal', 1, 100, 51)
    petal_width = st.sidebar.slider('Lebar Petal', 1, 100, 18)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = iris_parameters()

sl = df['sepal_length'].values[0]
sw = df['sepal_width'].values[0] 

pl = df['petal_length'].values[0]
pw = df['petal_width'].values[0]

st.write("SEPAL: Panjang = " + str(sl)+ " dan Lebar = "+str(sw))
st.write("PETAL: Panjang = " + str(pl)+ " dan Lebar = "+str(pw))

model = st.sidebar.selectbox("Pilihan model?", 
    ('Decision Tree','Gaussian Naive Bayes','Linear Discriminant Analysis'))

st.write("Pilihan Model: " +model)

url = "https://raw.githubusercontent.com/anakUGM/iris/master/iris-data.csv"
data = pd.read_csv(url)

train, test = train_test_split(data, test_size = 0.2, stratify = data['Result'], random_state = 42)

X_train = train[['SL','SW','PL','PW']]
y_train = train.Result
X_test = test[['SL','SW','PL','PW']]
y_test = test.Result

if model == "Decision Tree":
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train,y_train)
    prediction=mod_dt.predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)
elif model == "Gaussian Naive Bayes":
    mod_gnb_all = GaussianNB()
    prediction = mod_gnb_all.fit(X_train, y_train).predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)
else:
    mod_lda_all = LinearDiscriminantAnalysis()
    prediction = mod_lda_all.fit(X_train, y_train).predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)