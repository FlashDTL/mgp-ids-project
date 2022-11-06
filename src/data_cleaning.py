import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import KNNImputer

def convert_range(x):
    NewValue = (((x-0)*(1+1))/(1-0))-1 #NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    return NewValue

def data_cleaning(df):
    df.dropna()
    drop_columns = ['instance_id', 'artist_name', 'track_name', 'obtained_date'] # Removing all non-predictive features
    df.drop(drop_columns, inplace=True, axis=1)
    df['loudness'] = df['loudness'].abs()
    df['mode'].replace('Major', 1, inplace=True)
    df['mode'].replace('Minor', 0, inplace=True)
    df['duration_ms'] = df['duration_ms'].replace(-1, np.NaN)
    df['tempo'] = df['tempo'].replace('?', np.NaN)
    columns_cat = ['key']
    df_cat = df[columns_cat]
    df = df.drop(columns_cat, axis=1)
    df_cat = pd.get_dummies(df_cat)
    df = pd.concat([df_cat, df], axis = 1)
    encoder_genre = preprocessing.LabelEncoder()
    df["music_genre"] = encoder_genre.fit_transform(df["music_genre"])
    
    df['acousticness'] = df['acousticness'].apply(convert_range)
    df['danceability'] = df['danceability'].apply(convert_range)
    df['energy'] = df['energy'].apply(convert_range)
    df['instrumentalness'] = df['instrumentalness'].apply(convert_range)
    df['liveness'] = df['liveness'].apply(convert_range)
    df['speechiness'] = df['speechiness'].apply(convert_range)
    df['valence'] = df['valence'].apply(convert_range)
    
    imputer = KNNImputer(n_neighbors=121)
    df[:] = imputer.fit_transform(df)
    return df

