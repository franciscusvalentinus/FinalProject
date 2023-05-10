import numpy as np
import pickle
import streamlit as st
import pandas as pd

# loading the saved model
df = pd.read_csv("http://franciscusvalentinus.web.app/dataset/diabetes_data.csv")
loaded_model = pickle.load(open('/Users/franciscusvalentinus/PycharmProjects/pythonProject/venv/trained_model.sav', 'rb'))

def main():
    # giving a title
    st.title('Aplikasi Prediksi Risiko Hipertensi')

    # body mass index
    height = st.number_input('Tinggi badan', 150)
    weight = st.number_input('Berat badan', 40)
    bmi = round(weight / height / height * 10000, 2)
    st.title(f'BMI : {bmi}')

    # hypertension
    Age = st.number_input("Usia", 20)
    if Age <= 24:
        Age = 1.0
    elif Age <= 29:
        Age = 2.0
    elif Age <= 34:
        Age = 3.0
    elif Age <= 39:
        Age = 4.0
    elif Age <= 44:
        Age = 5.0
    elif Age <= 49:
        Age = 6.0
    elif Age <= 54:
        Age = 7.0
    elif Age <= 59:
        Age = 8.0
    elif Age <= 64:
        Age = 9.0
    elif Age <= 69:
        Age = 10.0
    elif Age <= 74:
        Age = 11.0
    elif Age <= 79:
        Age = 12.0
    else:
        Age = 13.0

    Sex = st.radio(
        "Jenis kelamin",
        ('Laki-laki', 'Perempuan'))
    if Sex == 'Laki-laki':
        Sex = 1.0
    else:
        Sex = 0.0

    HighChol = st.radio(
        "Apakah anda mempunyai kolesterol tinggi?",
        ('Iya', 'Tidak'))
    if HighChol == 'Iya':
        HighChol = 1.0
    else:
        HighChol = 0.0

    CholCheck = st.radio(
        "Apakah anda pernah melakukan cek kolesterol selama 5 tahun terakhir?",
        ('Iya', 'Tidak'))
    if CholCheck == 'Iya':
        CholCheck = 1.0
    else:
        CholCheck = 0.0

    BMI = st.slider('BMI', 12.0, max(df["BMI"]), bmi, disabled=True)

    Smoker = st.radio(
        "Apakah anda pernah merokok sebanyak 100 batang rokok selama hidup anda?",
        ('Iya', 'Tidak'))
    if Smoker == 'Iya':
        Smoker = 1.0
    else:
        Smoker = 0.0

    HeartDiseaseorAttack = st.radio(
        "Apakah anda pernah mengidap penyakit jantung koroner?",
        ('Iya', 'Tidak'))
    if HeartDiseaseorAttack == 'Iya':
        HeartDiseaseorAttack = 1.0
    else:
        HeartDiseaseorAttack = 0.0

    PhysActivity = st.radio(
        "Apakah anda melakukan aktivitas fisik dalam 30 hari terakhir (tidak termasuk pekerjaan)?",
        ('Iya', 'Tidak'))
    if PhysActivity == 'Iya':
        PhysActivity = 1.0
    else:
        PhysActivity = 0.0

    Fruits = st.radio(
        "Apakah anda mengkonsumsi buah 1 kali atau lebih setiap hari?",
        ('Iya', 'Tidak'))
    if Fruits == 'Iya':
        Fruits = 1.0
    else:
        Fruits = 0.0

    Veggies = st.radio(
        "Apakah anda mengkonsumsi sayur 1 kali atau lebih setiap hari?",
        ('Iya', 'Tidak'))
    if Veggies == 'Iya':
        Veggies = 1.0
    else:
        Veggies = 0.0

    HvyAlcoholConsump = st.radio(
        "Apakah anda mengkonsumsi alkohol (pria sebanyak 14 botol atau lebih, wanita sebanyak 7 botol atau lebih setiap minggu)?",
        ('Iya', 'Tidak'))
    if HvyAlcoholConsump == 'Iya':
        HvyAlcoholConsump = 1.0
    else:
        HvyAlcoholConsump = 0.0

    GenHlth = st.radio(
        "Bagaimana anda mengatakan kesehatan anda saat ini?",
        ('Luar biasa', 'Sangat baik', 'Baik', 'Biasa', 'Buruk'))
    if GenHlth == 'Luar biasa':
        GenHlth = 1.0
    elif GenHlth == 'Sangat baik':
        GenHlth = 2.0
    elif GenHlth == 'Baik':
        GenHlth = 3.0
    elif GenHlth == 'Biasa':
        GenHlth = 4.0
    else:
        GenHlth = 5.0

    MentHlth = float(st.number_input('Berapa hari dalam 1 bulan anda mengalami gangguan kesehatan mental?', 0))

    PhysHlth = float(st.number_input('Berapa hari dalam 1 bulan anda mengalami cedera fisik?', 0))

    DiffWalk = st.radio(
        "Apakah anda mengalami kesulitan saat berjalan atau menaiki tangga?",
        ('Iya', 'Tidak'))
    if DiffWalk == 'Iya':
        DiffWalk = 1.0
    else:
        DiffWalk = 0.0

    Stroke = st.radio(
        "Apakah anda pernah mengidap penyakit stroke?",
        ('Iya', 'Tidak'))
    if Stroke == 'Iya':
        Stroke = 1.0
    else:
        Stroke = 0.0

    Diabetes = st.radio(
        "Apakah anda mengidap penyakit diabetes?",
        ('Iya', 'Tidak'))
    if Diabetes == 'Iya':
        Diabetes = 1.0
    else:
        Diabetes = 0.0

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction
    if st.button('Submit'):
        diagnosis = hypertension_prediction(
            [Age, Sex, HighChol, CholCheck, BMI, Smoker, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Stroke, Diabetes])
    st.success(diagnosis)

# creating a function for Prediction
def hypertension_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'Orang ini tidak mengidap penyakit hipertensi'
    else:
        return 'Orang ini mengidap penyakit hipertensi'

if __name__ == '__main__':
    main()