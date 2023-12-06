# app.py
import streamlit as st
import joblib
import pandas as pd
import os
# from clases_y_funciones import selected_features, feat_eng, CustomQuantileTransformer, CustomStandardScaler, KMeansTransformer, input_features

input_features_num =    [ 
                    'MinTemp',
                    'MaxTemp',
                    'Rainfall',
                    'Evaporation',
                    'Sunshine',
                    'WindGustSpeed',
                    'WindSpeed9am',
                    'WindSpeed3pm',
                    'Humidity9am',
                    'Humidity3pm',
                    'Pressure9am',
                    'Pressure3pm',
                    'Cloud9am',
                    'Cloud3pm',
                    'Temp9am',
                    'Temp3pm',
                    ]

input_features_str =    [    
                        'Location',
                        'WindGustDir',
                        'WindDir9am',
                        'WindDir3pm',
                        'RainToday',
                        ]

input_features_date = ['Date']


st.title('Rain Predictor Model')

# Página principal para elegir el tipo de predicción
st.header("Choose Prediction Type")

if st.button("Predict Rainfall Amount"):
    prediction_type = "Rainfall Amount"

if st.button("Predict Rain or Not"):
    prediction_type = "Rain or Not"
       
# Lógica para la selección del tipo de predicción
if "prediction_type" in locals():
    st.subheader(f"Prediction Type: {prediction_type}")

    # Página para ingresar valores
    st.header("Enter Input Values")

    input_dict = {}

    with st.form(key='my_form'):
        for feat in input_features_num:
            input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
            input_dict[feat] = input_value
        for feat in input_features_str:
            input_value = st.text_input(f"Enter value for {feat}", value = "None")
            input_dict[feat] = input_value
        for feat in input_features_date:
            input_value = st.text_input("Select a date", value="yyyy-mm-dd")
            input_dict[feat] = input_value
            
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        st.write("Executing Prediction...")

        # Lógica para predecir según el tipo seleccionado
        if prediction_type == "Rainfall Amount":
            # Lógica para predecir cantidad de lluvia
            st.write("Executing Rainfall Amount Prediction...")
            # Agregar aquí la lógica para predecir la cantidad de lluvia
            pipe = joblib.load("rainfall_amount_prediction.pkl")
            user_input = pd.DataFrame(input_dict, index=[0])
            prediction = pipe.predict(user_input)
            prediction_value = prediction[0]
            st.header("Predicted Rainfall Amount")
            st.write(prediction_value)

        elif prediction_type == "Rain or Not":
            # Lógica para predecir si llueve o no
            st.write("Executing Rain or Not Prediction...")
            # Agregar aquí la lógica para predecir si llueve o no
            pipe = joblib.load("rain_or_not_prediction.pkl")
            user_input = pd.DataFrame(input_dict, index=[0])
            prediction = pipe.predict(user_input)
            prediction_value = prediction[0]
            st.header("Predicted Rain or Not")
            st.write("Mañana llueve" if prediction_value==1 else "Mañana no llueve")

    

st.markdown(
    """
    Rain prediction<br>
    Final Project<br>
    You can see the code in this [GitHub repository](https://github.com/augustofarias2/tp_apat)<br>
    """, unsafe_allow_html=True
)


