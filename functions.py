#Data handling
import pandas as pd
import numpy as np

#Data preprocessing
import joblib

#Data modeling
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from keras.models import load_model
import warnings

warnings.filterwarnings('ignore')


input_features =[
    'Date',
    'Location',
    'MinTemp',
    'MaxTemp',
    'Rainfall',
    'Evaporation',
    'Sunshine',
    'WindGustDir',
    'WindGustSpeed',
    'WindDir9am',
    'WindDir3pm',
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
    'RainToday',
    'RainTomorrow',
    'RainfallTomorrow'
    ]

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


def feat_eng_regr(df):
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
    
    df=vector_coordinates(df,'Wind9am','WindDir9am','WindSpeed9am')
    df=vector_coordinates(df,'Wind3pm','WindDir3pm','WindSpeed3pm')
    df=vector_coordinates(df,'WindGust','WindGustDir','WindGustSpeed')
    df['Max_Pressure'] = df[['Pressure9am', 'Pressure3pm']].max(axis=1)
    df['Min_Pressure'] = df[['Pressure9am', 'Pressure3pm']].min(axis=1)
    df['Max_Humidity'] = df[['Humidity9am', 'Humidity3pm']].max(axis=1)
    df['Min_Humidity'] = df[['Humidity9am', 'Humidity3pm']].min(axis=1)
    df['Estado_Cielo'] = df.apply(definir_estado_del_cielo, axis=1)
    df['Cluster'] = cluster_cities(df)

    df.drop(['Cloud3pm','Cloud9am','Humidity3pm','Humidity9am','WindSpeed9am','WindSpeed3pm','WindGustSpeed','WindDir9am','WindDir3pm','WindGustDir','Pressure3pm','Pressure9am','Temp3pm','Temp9am'], axis=1, inplace=True)
    df = df.replace([np.inf, -np.inf], 0)

    return df


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
    df_normalized['RainTomorrow'] = df_normalized['RainTomorrow'].map({'No': 0, 'Yes': 1})
    df_normalized['Date'] = pd.to_datetime(df_normalized['Date'])
    df_normalized['Date']=df_normalized['Date'].dt.dayofyear
    df_normalized['Estado_Cielo'] = df_normalized['Estado_Cielo'].map(skystate)
    scaler = StandardScaler()
    df_normalized[features] = scaler.fit_transform(df_normalized[features])
    
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
    ('Feature Engineering', FunctionTransformer(feat_eng_clas)),
    ('Model', clasificadorNN)
])
joblib.dump(pipeline_clasf, 'rain_or_not_prediction.joblib')



regresionNN = load_model('RegresionNN.h5')
pipeline_regress = Pipeline([
    ('Feature Engineering', FunctionTransformer(feat_eng_regr)),
    ('df Normalized', FunctionTransformer(dataframe_normalized_regress, kw_args={'features': selected_features})),
    ('Model', regresionNN)
])
joblib.dump(pipeline_regress, 'rainfall_amount_prediction.joblib')
