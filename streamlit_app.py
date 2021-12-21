import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from seaborn.axisgrid import pairplot
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree

#Datasets 
datasets_url = "https://raw.githubusercontent.com/anakUGM/iris/master/iris-data.csv"
#Image Resources 
iris_setosa_url = "https://github.com/anakUGM/iris/raw/master/iris-setosa.jpg"
iris_virginica_url = "https://github.com/anakUGM/iris/raw/master/iris-virginica.jpg"
iris_versicolor_url = "https://github.com/anakUGM/iris/raw/master/iris-versicolor.jpg"
pair_plot_url = "https://github.com/anakUGM/iris/raw/master/pair-plot.png"

#Functions 
def iris_parameters():
    sl = st.sidebar.slider('Panjang Sepal (mm)', 40, 85, 59)
    sw = st.sidebar.slider('Lebar Sepal (mm)', 15, 50, 30)
    pl = st.sidebar.slider('Panjang Petal (mm)', 8, 75, 50)
    pw = st.sidebar.slider('Lebar Petal (mm)', 1, 30, 18)
    data = {'sl': sl,
            'sw': sw,
            'pl': pl,
            'pw': pw}
    features = pd.DataFrame(data, index=[0])
    return features

def datasets_information(st):
    st.write("""
    ## Deskripsi Data
    Informasi mengenai dataset yang dipakai.""")
    st.write("""### Statistik Dataset""")
    st.table(data.describe())
    st.write("""### Pair Plot""")
    #pp = sns.pairplot(train, hue="Result", height = 2, palette = 'colorblind')
    #st.pyplot(pp)
    #Ganti dengan statik image supaya bisa lebih cepat load 
    st.image(pair_plot_url)

def prediction_result(st, result):
    st.write("Hasil Prediksi: "+ result)
    if result=="Iris-virginica":
        st.image(iris_virginica_url)
    elif result=="Iris-setosa":
        st.image(iris_setosa_url)
    else:
        st.image(iris_versicolor_url)

def algorithm_decission_tree(x_train, y_train, x_test, y_test, df, st, fn, cn):
    dt_classifier = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    prediction = dt_classifier.fit(x_train,y_train).predict(df)
    prediction_result(st, prediction[0])
    datasets_information(st)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai algoritma klasifikasi yang dipakai.
    ### Decision Tree 
    """)

    dt_classifier.fit(x_train,y_train)
    prediction=dt_classifier.predict(x_test)
    y_pred = dt_classifier.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    st.write("""#### Contoh Tes Data""")
    st.table(x_test)
    st.write("""#### Contoh Hasil Tes""")
    st.table(y_test)
    st.write("""#### Report""")
    st.code(report)
    st.write("Confusion Matrix")
    metrics.plot_confusion_matrix(dt_classifier, x_test, y_test,
        display_labels=cn, cmap=plt.cm.Blues, normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Decission Tree adalah: "+str(round(accuracy, 4)))
    st.write("""#### Visualisasi """)
    plt.figure(figsize = (10,8))
    plot_tree(dt_classifier, feature_names = fn, class_names = cn, filled = True)
    st.pyplot(plt)

def algorithm_gausian_naive_bayes(x_train, y_train, x_test, y_test, df, st, fn, cn):
    gnb_classifier = GaussianNB()
    prediction = gnb_classifier.fit(x_train, y_train).predict(df)
    prediction_result(st, prediction[0])
    datasets_information(st)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai algoritma klasifikasi yang dipakai.
    ### Gaussian Naive Bayes
    """)

    gnb_classifier = GaussianNB()
    prediction = gnb_classifier.fit(x_train, y_train).predict(x_test)
    y_pred = gnb_classifier.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    st.write("""#### Contoh Tes Data""")
    st.table(x_test)
    st.write("""#### Contoh Hasil Tes""")
    st.table(y_test)
    st.write("""#### Report""")
    st.code(report)
    st.write("Confusion Matrix")
    metrics.plot_confusion_matrix(gnb_classifier, x_test, y_test,
        display_labels=cn,cmap=plt.cm.Blues,normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Gaussian Naive Bayes adalah: "+str(round(accuracy, 4)))

def algorithm_linear_discriminant_analysis(x_train, y_train, x_test, y_test, df, st, fn, cn):
    lda_classifier = LinearDiscriminantAnalysis()
    prediction = lda_classifier.fit(x_train, y_train).predict(df)
    prediction_result(st, prediction[0])
    datasets_information(st)

    st.write("""
    ## Model Klasifikasi
    Informasi mengenai algoritma klasifikasi yang dipakai.
    ### Linear Discriminant Analysis
    """)

    prediction = lda_classifier.fit(x_train, y_train).predict(x_test)
    y_pred = lda_classifier.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_pred, y_test)
    st.write("""#### Contoh Tes Data""")
    st.table(x_test)
    st.write("""#### Contoh Hasil Tes""")
    st.table(y_test)
    st.write("""#### Report""")
    st.code(report)
    st.write("Confusion Matrix")
    metrics.plot_confusion_matrix(lda_classifier, x_test, y_test,
        display_labels=cn, cmap=plt.cm.Blues, normalize=None)
    st.pyplot(plt)
    st.write("Akurasi Linear Discriminant Analysis adalah: "+str(round(accuracy, 4)))

#UI 
st.write("""
# Prediksi Jenis Bunga Iris
Aplikasi ini akan memprediksi jenis bunga iris berdasarkan nilai parameter Sepal dan Petal.
""")
st.write("1. Masukan nilai Sepal dan Petal dengan menggeser ke nilai yang sesuai.")
st.write("2. Pilih algoritma untuk melakukan prediksi lewat combo box yang ada.")
st.sidebar.header('Parameter Sepal dan Petal')
df = iris_parameters()
sl = round(df['sl'].values[0] * 0.1, 2)
sw = round(df['sw'].values[0] * 0.1, 2)
pl = round(df['pl'].values[0] * 0.1, 2)
pw = round(df['pw'].values[0] * 0.1, 2)
st.write("SEPAL: Panjang = " + str(sl)+ " cm dan Lebar = "+str(sw) + " cm")
st.write("PETAL: Panjang = " + str(pl)+ " cm dan Lebar = "+str(pw) + " cm")
algorithm = st.sidebar.selectbox("Pilihan algoritma?", 
    ('Decision Tree','Gaussian Naive Bayes','Linear Discriminant Analysis'))
st.write("Pilihan Algoritma: " +algorithm)

#Main Function 
data = pd.read_csv(datasets_url)
train, test = train_test_split(data, test_size = 0.1, stratify = data['Result'], random_state = 42)
x_train = train[['SL','SW','PL','PW']]
y_train = train.Result
x_test = test[['SL','SW','PL','PW']]
y_test = test.Result
fn = ['SL', 'SW', 'PL', 'PW']
cn = ['setosa', 'versicolor', 'virginica']

if algorithm == "Decision Tree":
    algorithm_decission_tree(x_train, y_train, x_test, y_test, df, st, fn, cn)
elif algorithm == "Gaussian Naive Bayes":
    algorithm_gausian_naive_bayes(x_train, y_train, x_test, y_test, df, st, fn, cn)
else:
    algorithm_linear_discriminant_analysis(x_train, y_train, x_test, y_test, df, st, fn, cn)