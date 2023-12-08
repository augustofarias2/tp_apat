# app.py
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta
from functions import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

def cluster_cities(df):
    city_cluster = {
        'Adelaide': 10,
        'Albany': 1,
        'Albury': 4,
        'AliceSprings': 3,
        'BadgerysCreek': 2,
        'Ballarat': 8,
        'Bendigo': 8,
        'Brisbane': 9,
        'Cairns': 6,
        'Canberra': 4,
        'Cobar': 4,
        'CoffsHarbour': 9,
        'Dartmoor': 0,
        'Darwin': 7,
        'GoldCoast': 9,
        'Hobart': 12,
        'Katherine': 7,
        'Launceston': 12,
        'Melbourne': 8,
        'MelbourneAirport': 8,
        'Mildura': 0,
        'Moree': 9,
        'MountGambier': 0,
        'MountGinini': 4,
        'Newcastle': 2,
        'Nhil': 0,
        'NorahHead': 2,
        'NorfolkIsland': 5,
        'Nuriootpa': 10,
        'PearceRAAF': 1,
        'Penrith': 2,
        'Perth': 1,
        'PerthAirport': 1,
        'Portland': 0,
        'Richmond': 2,
        'Sale': 8,
        'Sydney': 2,
        'SydneyAirport': 2,
        'Townsville': 6,
        'Tuggeranong': 4,
        'Uluru': 3,
        'WaggaWagga': 4,
        'Walpole': 1,
        'Watsonia': 8,
        'Williamtown': 2,
        'Witchcliffe': 1,
        'Wollongong': 2,
        'Woomera': 10
        }

    cluster = df['Location'].map(city_cluster)

    return cluster

def vector_coordinates(df, column, dir, speed, threshold=1e-6):
    direcciones = {
        'N': 0,
        'NNE': 22.5,
        'NE': 45,
        'ENE': 67.5,
        'E': 90,
        'ESE': 112.5,
        'SE': 135,
        'SSE': 157.5,
        'S': 180,
        'SSW': 202.5,
        'SW': 225,
        'WSW': 247.5,
        'W': 270,
        'WNW': 292.5,
        'NW': 315,
        'NNW': 337.5
    }
    velocidad = df[speed]
    df[column + 'u'] = -velocidad * np.sin(np.deg2rad(df[dir].map(direcciones)))
    df[column + 'v'] = -velocidad * np.cos(np.deg2rad(df[dir].map(direcciones)))
    
    # Comprobación condicional para valores cercanos a cero
    df[column + 'u'][np.abs(df[column + 'u']) < threshold] = 0
    df[column + 'v'][np.abs(df[column + 'v']) < threshold] = 0
    
    return df

def definir_estado_del_cielo(row):
    if row['Cloud9am'] == 0 and row['Cloud3pm'] == 0:
        return 'Despejado'
    elif row['Cloud9am'] <= 4 and row['Cloud3pm'] <= 4:
        return 'Parcialmente Nublado'
    elif row['Cloud9am'] <= 7 and row['Cloud3pm'] <= 7:
        return 'Nublado'
    else:
        return 'Muy Nublado'


def feat_eng_clas(df):
    df_normalized = df.copy()
    df_normalized['Min_Humidity'] = df_normalized[['Humidity9am', 'Humidity3pm']].min(axis=1)

    df_normalized.drop(['Humidity3pm','Humidity9am'], axis=1, inplace=True)
    df_normalized = df_normalized.replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_normalized), columns=df_normalized.columns)

    X = df_normalized[['Sunshine','Min_Humidity']]

    return X

def dataframe_normalized_regress(df,features):
    df_normalized = df.copy()
    skystate = {
        'Despejado' :               0,
        'Parcialmente Nublado' :    1,
        'Nublado' :                 2,
        'Muy Nublado' :             3
    }
    df_normalized['RainToday'] = df_normalized['RainToday'].map({'No': 0, 'Yes': 1})
    df_normalized['Date'] = pd.to_datetime(df_normalized['Date'])
    df_normalized['Date']=df_normalized['Date'].dt.dayofyear
    df_normalized['Estado_Cielo'] = df_normalized['Estado_Cielo'].map(skystate)

    columns =   [
                    'MinTemp', 
                    'MaxTemp', 
                    'Evaporation',
                    'Sunshine',
                    'Wind9amu', 
                    'Wind9amv', 
                    'Wind3pmu', 
                    'Wind3pmv', 
                    'WindGustu',
                    'WindGustv', 
                    'Max_Pressure',
                    'Min_Pressure',
                    'Max_Humidity',
                    'Min_Humidity',
                    'Estado_Cielo',
                    'Cluster',
                    'Date'
                ]

    X = df_normalized[columns]

    return X




clasificadorNN = load_model('ClasificadorNN.h5')
pipeline_clasf = Pipeline([
    # ('Feature Engineering', FunctionTransformer(feat_eng_clas)),
    ('Model', clasificadorNN)
])


regresionNN = load_model('RegresionNN.h5')
pipeline_regress = Pipeline([
    # ('Feature Engineering', FunctionTransformer(feat_eng_regr)),
    # ('df Normalized', FunctionTransformer(dataframe_normalized_regress, kw_args={'features': selected_features})),
    ('Model', regresionNN)
])


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


def predict_rainfall_amount(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0][0]

def predict_rain_or_not(model, user_input):
    prediction = model.predict(user_input)
    return prediction[0][0]

st.title('Rain Predictor Model')


def get_user_input(input_features_num, input_features_str, input_features_date):
    
    columns = [
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
        'Location',
        'WindGustDir',
        'WindDir9am',
        'WindDir3pm',
        'RainToday',
        'Date'
    ]

    # Página para ingresar valores
    st.header("Enter Input Values")

    input_dict = {}

    with st.form(key="my_form"):
    # if input_features_num is not None:
        for feat in input_features_num:
            input_value = st.number_input(f"Enter value for {feat}", value=0.0, step=0.01)
            input_dict[feat] = input_value
    
    # if input_features_str is not None:
        for feat in input_features_str:
            input_value = st.text_input(f"Enter value for {feat}", value = "None")
            input_dict[feat] = input_value

    # if input_features_date is not None:
        for feat in input_features_date:
            today = datetime.today()
            one_year_ago = today - timedelta(days=10000)
            input_value = st.date_input("Select a date", min_value=one_year_ago)
            formatted_date = input_value.strftime("%Y-%m-%d")
            input_dict[feat] = formatted_date
            
        submit_button = st.form_submit_button(label='Submit')

    user_input = pd.DataFrame(input_dict, index=[0], columns=columns)

    user_input=vector_coordinates(user_input,'Wind9am','WindDir9am','WindSpeed9am')
    user_input=vector_coordinates(user_input,'Wind3pm','WindDir3pm','WindSpeed3pm')
    user_input=vector_coordinates(user_input,'WindGust','WindGustDir','WindGustSpeed')
    user_input['Max_Pressure'] = user_input[['Pressure9am', 'Pressure3pm']].max(axis=1)
    user_input['Min_Pressure'] = user_input[['Pressure9am', 'Pressure3pm']].min(axis=1)
    user_input['Max_Humidity'] = user_input[['Humidity9am', 'Humidity3pm']].max(axis=1)
    user_input['Min_Humidity'] = user_input[['Humidity9am', 'Humidity3pm']].min(axis=1)
    user_input['Estado_Cielo'] = user_input.apply(definir_estado_del_cielo, axis=1)
    user_input['Cluster'] = cluster_cities(user_input)

    selected_features = [
                    'MinTemp', 
                    'MaxTemp', 
                    'Evaporation',
                    'Sunshine',
                    'Wind9amu', 
                    'Wind9amv', 
                    'Wind3pmu', 
                    'Wind3pmv', 
                    'WindGustu',
                    'WindGustv', 
                    'Max_Pressure',
                    'Min_Pressure',
                    'Max_Humidity',
                    'Min_Humidity',
                    'Estado_Cielo',
                    'Cluster',
                    'Date'
                ]

    user_input=dataframe_normalized_regress(user_input,selected_features)
    # user_input.drop(['Cloud3pm','Cloud9am','Humidity3pm','Humidity9am','WindSpeed9am','WindSpeed3pm','WindGustSpeed','WindDir9am','WindDir3pm','WindGustDir','Pressure3pm','Pressure9am','Temp3pm','Temp9am'], axis=1, inplace=True)
    
    # input_array = user_input.iloc[0].to_dict()

    return user_input, submit_button

user_input, submit_button = get_user_input(input_features_num, input_features_str, input_features_date)


if submit_button:
    st.write(user_input)
    st.write("Executing Rain or Not Prediction...")
    prediction_value = predict_rain_or_not(pipeline_clasf, user_input[['Sunshine', 'Min_Humidity']])
    # st.header("Predicted Rain or Not:")
    st.header("Mañana llueve" if prediction_value >= 0.5 else "Mañana no llueve")

    if prediction_value >= 0.5:
        st.write("Executing Rainfall Amount Prediction...")
        prediction_value = predict_rainfall_amount(pipeline_regress, user_input)
        # st.header("Predicted Rainfall Amount:")
        st.header(f"Lloverán aproximadamente [{prediction_value}] mm")
    

st.markdown(
    """
    Final Project of rain prediction<br>
    You can see the code in this [GitHub repository](https://github.com/augustofarias2/tp_apat)<br>
    """, unsafe_allow_html=True
)