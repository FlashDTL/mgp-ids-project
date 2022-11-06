# Functions for preprocessing (data cleaning + scaling) and feature selection

import pandas as pd

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_train(df_train):
    """
    Preprocessing + feature selection for training data.

    Returns preprocessed dataframe and fitted imputers/scalers required for test data.
    """

    # Drop unneeded features
    df_train = df_train.drop(['instance_id', 'artist_name', 'track_name', 'obtained_date'], axis=1)

    # Drop missing values
    df_train = df_train.dropna()                       # -4 samples
    df_train = df_train[df_train['loudness'] != 0]     # -1 sample

    # Impute missing values with means in respective columns (affects 4012+3560 samples)
    #df_train.loc[df_train['tempo'] == '?', 'tempo'] = -1    # temporarily substitute '?'s with '-1's
    #imputer_nums = SimpleImputer(
    #    missing_values = -1,
    #    strategy = 'mean'
    #)
    #df_train[['tempo', 'duration_ms']] = imputer_nums.fit_transform(df_train[['tempo', 'duration_ms']])
    
    # Old imputing strategy - drop samples with missing values
    df_train = df_train[df_train['tempo'] != '?']           # -4012 samples
    df_train = df_train[df_train['duration_ms'] != -1]      # -3560 samples
    imputer_nums = None

    # Convert tempo dtype from 'object' to 'float64' (type was 'object' because of presence of '?' in data)
    df_train['tempo'] = df_train['tempo'].astype('float64')

    # Separate categorical features
    columns_cat = ['key', 'mode']
    df_train_cat = df_train[columns_cat]
    df_train = df_train.drop(columns_cat, axis=1)

    # One-hot
    df_train_cat = pd.get_dummies(df_train_cat)

    # Concatenate categorical columns with others
    df_train = pd.concat([df_train, df_train_cat], axis=1)

    # Encode output categories into integers
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train['music_genre'])
    df_train['music_genre'] = label_encoder.transform(df_train['music_genre'])

    # Transform loudness from db into power (take logarithm base 10)
    #df_train['loudness'] = df_train['loudness'].apply(lambda x : x + 60)
    #df_train['loudness'] = df_train['loudness'].apply(lambda x : x / 10)
    #df_train['loudness'] = df_train['loudness'].apply(lambda x : np.power(10, x))

    # Get polynomical features
    cols_float = df_train.columns[df_train.dtypes == 'float64']
    df_train_float = df_train[cols_float]
    poly_features = PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )
    df_train_float = poly_features.fit_transform(df_train_float)
    cols_poly = poly_features.get_feature_names_out(cols_float)
    df_train_float = pd.DataFrame(df_train_float, columns=cols_poly, index=df_train.index)

    # Scale features
    std_scaler = StandardScaler()
    df_train_float[:] = std_scaler.fit_transform(df_train_float)

    # Put floats/polynomials back into original dataframe
    df_train = df_train.drop(cols_float, axis=1)
    df_train = pd.concat((df_train_float, df_train), axis=1)

    # Dictionary to return
    res_dict = {
        'dataframe' : df_train,
        'utils' : [label_encoder, imputer_nums, std_scaler]
    }

    return res_dict

def preprocess_test(df_test, utils):
    """
    Preprocessing + feature selection for test data. Uses utils fitted on training data.

    Returns preprocessed dataframe.
    """

    # Unpack utils
    label_encoder = utils[0]
    imputer_nums = utils[1]
    std_scaler = utils[2]

    # Drop unneeded features
    df_test = df_test.drop(['instance_id', 'artist_name', 'track_name', 'obtained_date'], axis=1)

    # Drop missing values
    df_test = df_test.dropna()
    df_test = df_test[df_test['loudness'] != 0]

    # Impute missing values with means in respective columns
    #df_test.loc[df_test['tempo'] == '?', 'tempo'] = -1    # temporarily substitute '?'s with '-1's
    #df_test[['tempo', 'duration_ms']] = imputer_nums.transform(df_test[['tempo', 'duration_ms']])

    # Old imputing strategy - drop samples with missing values
    df_test= df_test[df_test['tempo'] != '?']
    df_test = df_test[df_test['duration_ms'] != -1]

    # Convert tempo dtype from 'object' to 'float64' (type was 'object' because of presence of '?' in data)
    df_test['tempo'] = df_test['tempo'].astype('float64')

    # Separate categorical features
    columns_cat = ['key', 'mode']
    df_test_cat = df_test[columns_cat]
    df_test = df_test.drop(columns_cat, axis=1)

    # One-hot
    df_test_cat = pd.get_dummies(df_test_cat)

    # Concatenate categorical columns with others
    df_test = pd.concat([df_test, df_test_cat], axis=1)

    # Encode output categories into integers
    # NB! this part uses label encoder fitted on training data
    label_encoder.transform(df_test['music_genre'])
    df_test['music_genre'] = label_encoder.transform(df_test['music_genre'])

    # Transform loudness from db into power (take logarithm base 10)
    #df_test['loudness'] = df_test['loudness'].apply(lambda x : x + 60)
    #df_test['loudness'] = df_test['loudness'].apply(lambda x : x / 10)
    #df_test['loudness'] = df_test['loudness'].apply(lambda x : np.power(10, x))

    # Get polynomical features
    cols_float = df_test.columns[df_test.dtypes == 'float64']
    df_test_float = df_test[cols_float]
    poly_features = PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )
    df_test_float = poly_features.fit_transform(df_test_float)
    cols_poly = poly_features.get_feature_names_out(cols_float)
    df_test_float = pd.DataFrame(df_test_float, columns=cols_poly, index=df_test.index)

    # Scale features
    # NB! this part uses scaler fitted on training data
    df_test_float[:] = std_scaler.transform(df_test_float)

    # Put floats/polynomials back into original dataframe
    df_test = df_test.drop(cols_float, axis=1)
    df_test = pd.concat((df_test_float, df_test), axis=1)

    # Return dataframe
    return df_test