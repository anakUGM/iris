from seaborn.axisgrid import pairplot
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Image Resources 
iris_setosa_url = "https://github.com/anakUGM/iris/raw/master/iris-setosa.jpg"
iris_virginica_url = "https://github.com/anakUGM/iris/raw/master/iris-virginica.jpg"
iris_versicolor_url = "https://github.com/anakUGM/iris/raw/master/iris-versicolor.jpg"
pair_plot_url = "https://github.com/anakUGM/iris/raw/master/pair-plot.png"

st.write("""
# Prediksi Jenis Bunga Iris
Aplikasi ini akan memprediksi jenis bunga iris berdasarkan nilai parameter Sepal dan Petal.
""")
st.write("1. Masukan nilai Sepal dan Petal dengan menggeser ke nilai yang sesuai.")
st.write("2. Pilih model untuk melakukan prediksi lewat combo box yang ada.")
st.sidebar.header('Parameter Sepal dan Petal')

def iris_parameters():
    sepal_length = st.sidebar.slider('Panjang Sepal (mm)', 40, 85, 59)
    sepal_width = st.sidebar.slider('Lebar Sepal (mm)', 15, 50, 30)
    petal_length = st.sidebar.slider('Panjang Petal (mm)', 8, 75, 50)
    petal_width = st.sidebar.slider('Lebar Petal (mm)', 1, 30, 18)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = iris_parameters()

sl = round(df['sepal_length'].values[0] * 0.1, 2)
sw = round(df['sepal_width'].values[0] * 0.1, 2)

pl = round(df['petal_length'].values[0] * 0.1, 2)
pw = round(df['petal_width'].values[0] * 0.1, 2)

st.write("SEPAL: Panjang = " + str(sl)+ " cm dan Lebar = "+str(sw) + " cm")
st.write("PETAL: Panjang = " + str(pl)+ " cm dan Lebar = "+str(pw) + " cm")

model = st.sidebar.selectbox("Pilihan model?", 
    ('Decision Tree','Gaussian Naive Bayes','Linear Discriminant Analysis'))

st.write("Pilihan Model: " +model)

url = "https://raw.githubusercontent.com/anakUGM/iris/master/iris-data.csv"
data = pd.read_csv(url)

train, test = train_test_split(data, test_size = 0.1, stratify = data['Result'], random_state = 42)

x_train = train[['SL','SW','PL','PW']]
y_train = train.Result
x_test = test[['SL','SW','PL','PW']]
y_test = test.Result


if model == "Decision Tree":
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    prediction = mod_dt.fit(x_train,y_train).predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)

    st.write("""
    ## Deskripsi Data
    Informasi mengenai dataset yang dipakai.
    """)
    st.write("""
    ### Statistik Dataset 
    """)
    st.write(data.describe())
    st.write("""
    ### Pair Plot
    """)
    #pp = sns.pairplot(train, hue="Result", height = 2, palette = 'colorblind')
    #st.pyplot(pp)
    #Ganti dengan statik image supaya bisa lebih cepat load 
    st.image(pair_plot_url)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai model klasifikasi yang dipakai.
    """)
    st.write("""
    ### Decision Tree
    """)
    x_train = train[['SL','SW','PL','PW']]
    y_train = train.Result
    x_test = test[['SL','SW','PL','PW']]
    y_test = test.Result
    
    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(x_train,y_train)
    prediction=mod_dt.predict(x_test)
    y_pred = mod_dt.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    cm = confusion_matrix(y_test, y_pred)

    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']

    st.write("""
    #### Contoh Tes Data
    """)
    st.table(x_test)
    st.write("""
    #### Contoh Hasil Tes
    """)
    st.table(y_test)
    st.write("""
    #### Report
    """)
    st.code(report)
    st.write("Confusion Matrix")
    #st.code(cm)
    disp = metrics.plot_confusion_matrix(mod_dt, x_test, y_test,
        display_labels=cn,cmap=plt.cm.Blues,normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Decission Tree adalah: "+str(round(accuracy, 4)))
    st.write("""
    #### Visualisasi 
    """)
    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']
    plt.figure(figsize = (10,8))
    plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True)
    st.pyplot(plt)
elif model == "Gaussian Naive Bayes":
    mod_gnb_all = GaussianNB()
    prediction = mod_gnb_all.fit(x_train, y_train).predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)
    
    st.write("""
    ## Deskripsi Data
    Informasi mengenai dataset yang dipakai.
    """)
    st.write("""
    ### Statistik Dataset 
    """)
    st.write(data.describe())
    st.write("""
    ### Pair Plot
    """)
    #pp = sns.pairplot(train, hue="Result", height = 2, palette = 'colorblind')
    #st.pyplot(pp)
    #Ganti dengan statik image supaya bisa lebih cepat load 
    st.image(pair_plot_url)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai model klasifikasi yang dipakai.
    """)
    st.write("""
    ### Gaussian Naive Bayes
    """)
    x_train = train[['SL','SW','PL','PW']]
    y_train = train.Result
    x_test = test[['SL','SW','PL','PW']]
    y_test = test.Result
    
    mod_gnb_all = GaussianNB()
    prediction = mod_gnb_all.fit(x_train, y_train).predict(x_test)
    y_pred = mod_gnb_all.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    cm = confusion_matrix(y_test, y_pred)

    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']

    st.write("""
    #### Contoh Tes Data
    """)
    st.table(x_test)
    st.write("""
    #### Contoh Hasil Tes
    """)
    st.table(y_test)
    st.write("""
    #### Report
    """)
    st.code(report)
    st.write("Confusion Matrix")
    disp = metrics.plot_confusion_matrix(mod_gnb_all, x_test, y_test,
        display_labels=cn,cmap=plt.cm.Blues,normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Gaussian Naive Bayes adalah: "+str(round(accuracy, 4)))
    #st.write("""
    #### Visualisasi 
    #""")
    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']
    #plt.figure(figsize = (10,8))
    #plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True)
    #st.pyplot(plt)

else:
    mod_lda_all = LinearDiscriminantAnalysis()
    prediction = mod_lda_all.fit(x_train, y_train).predict(df)
    st.write("Hasil Prediksi: "+ prediction[0])
    if prediction[0]=="Iris-virginica":
        st.image(iris_virginica_url)
    elif prediction[0]=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)
    
    st.write("""
    ## Deskripsi Data
    Informasi mengenai dataset yang dipakai.
    """)
    st.write("""
    ### Statistik Dataset 
    """)
    st.write(data.describe())
    st.write("""
    ### Pair Plot
    """)
    #pp = sns.pairplot(train, hue="Result", height = 2, palette = 'colorblind')
    #st.pyplot(pp)
    #Ganti dengan statik image supaya bisa lebih cepat load 
    st.image(pair_plot_url)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai model klasifikasi yang dipakai.
    """)
    st.write("""
    ### Linear Discriminant Analysis
    """)
    x_train = train[['SL','SW','PL','PW']]
    y_train = train.Result
    x_test = test[['SL','SW','PL','PW']]
    y_test = test.Result
    
    mod_lda_all = LinearDiscriminantAnalysis()
    prediction = mod_lda_all.fit(x_train, y_train).predict(x_test)
    y_pred = mod_lda_all.predict(x_test)

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    cm = confusion_matrix(y_test, y_pred)

    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']

    st.write("""
    #### Contoh Tes Data
    """)
    st.table(x_test)
    st.write("""
    #### Contoh Hasil Tes
    """)
    st.table(y_test)
    st.write("""
    #### Report
    """)
    st.code(report)
    st.write("Confusion Matrix")
    disp = metrics.plot_confusion_matrix(mod_lda_all, x_test, y_test,
        display_labels=cn,cmap=plt.cm.Blues,normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Linear Discriminant Analysis adalah: "+str(round(accuracy, 4)))
    #st.write("""
    #### Visualisasi 
    #""")
    fn = ['SL', 'SW', 'PL', 'PW']
    cn = ['setosa', 'versicolor', 'virginica']
    #plt.figure(figsize = (10,8))
    #plot_tree(mod_dt, feature_names = fn, class_names = cn, filled = True)
    #st.pyplot(plt)
