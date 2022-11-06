# Functions for training neural networks using Keras.

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def build_nn_model(
        optimizer = 'adam',
        reg_rate_l2 = 0.001,
        reg_rate_l1 = 0,
        kernel_init = 'glorot_uniform',
        activation = 'relu',
        hidden_units_in_layer = 10,
        hidden_layers = 3
        ):
    """
    Makes and compiles a NN model with given hyperparameters and (partially) structure.
    
    Returns sequential Keras model.
    """
    
    # Sequential model
    model_nn = keras.models.Sequential()
    
    # Add hidden layers
    for i in range(hidden_layers):
        model_nn.add(
            layers.Dense(
                units = hidden_units_in_layer,
                activation = activation,
                kernel_regularizer = keras.regularizers.l2(reg_rate_l2),
                bias_regularizer = keras.regularizers.l2(reg_rate_l2),
                activity_regularizer = keras.regularizers.l2(reg_rate_l2),
                kernel_initializer = kernel_init
            )
        )
    
    # Add softmax layer
    model_nn.add(
        layers.Dense(units=10)
    )
    model_nn.add(
        layers.Softmax()
    )
    
    # Compile model
    model_nn.compile(
        optimizer = optimizer,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model_nn

def cross_validate_nn(
        X_train_val,
        y_train_val,
        optimizer = 'adam',
        reg_rate_l2 = 0.001,
        activation = 'relu',
        hidden_units_in_layer = 50,
        hidden_layers = 3,
        batch_size = 32,
        epochs = 2,
        verbose = True
        ):
    """
    Performs 4-fold cross-validation on Keras model with given hyperparameters.

    Returns model, accuracy, history (Keras) and a string of used hyperparameters.
    """

    # List of results
    res_grid_cv = {
        'models' : [],
        'loss_metrics' : [],
        'history' : []
    }
    
    # Current fold index
    ind_fold = 1

    # K-fold cross-validation (4 folds)
    kfold = KFold(
        n_splits = 4,
        shuffle = False
    )
    for train_index, val_index in kfold.split(X_train_val):
        
        # Report info (verbose)
        if verbose:
            print("FOLD:", ind_fold, "/", kfold.get_n_splits())
        
        # Update fold index
        ind_fold += 1

        # Split data
        X_train = X_train_val.iloc[train_index]
        X_val = X_train_val.iloc[val_index]
        y_train = y_train_val.iloc[train_index]
        y_val = y_train_val.iloc[val_index]

        # Prepare model
        model_nn = build_nn_model()

        # Train model
        hist = model_nn.fit(
            x = X_train,
            y = y_train,
            batch_size = batch_size,
            epochs = epochs,
            #validation_split = 0.25, # overridden by validation_data
            validation_data = (X_val, y_val)
        )

        # Measure performance (accuracy)
        val_loss_acc_final = model_nn.evaluate(X_val, y_val)

        # Append results to return
        res_grid_cv['models'].append(model_nn)
        res_grid_cv['loss_metrics'].append(val_loss_acc_final)
        res_grid_cv['history'].append(hist)

    # Find the best model (best fold)
    best_ind = 0
    best_acc = 0
    for i in range(kfold.get_n_splits()):
        acc = res_grid_cv['loss_metrics'][i][1]
        if acc > best_acc:
            best_ind = i
            best_acc = acc
    
    # Save parameters for record
    params = "opt:" + str(optimizer) + ", L2:" + str(reg_rate_l2) + ", act:" + str(activation) + \
        ", hid_dim:" + str(hidden_units_in_layer) + ", hid_n:" + str(hidden_layers) + \
        ", batch_size:" + str(batch_size) + ", epochs:" + str(epochs)

    # Return the best model and info on it
    res_best = {
        'model' : res_grid_cv['models'][i],
        'val_accuracy' : best_acc,
        'hist' : res_grid_cv['history'][i],
        'params' : params
    }
    
    return res_best

def grid_search_cv_nn(X_train_val, y_train_val, param_grid, verbose = True):
    """
    Performs grid search of given hyperparameters, creating model for each 
    combination and doing 4-fold cross validation for it.
    """
    
    # Extract parameters
    optimizer_arr = param_grid['optimizer']
    reg_rate_l2_arr = param_grid['reg_rate_l2']
    activation_arr = param_grid['activation']
    hidden_dim_arr = param_grid['hidden_dim']
    hidden_layers_arr = param_grid['hidden_layers']
    batch_size_arr = param_grid['batch_size']
    epochs_arr = param_grid['epochs']
    
    # Extract parameters
    #optimizer_arr = ['adam', 'rmsprop']
    #reg_rate_l2_arr = [0.0001, 0.0003, 0.001, 0.003, 0.007]
    #activation_arr = ['relu', 'tanh']
    #hidden_dim_arr = [30, 50, 100]
    #hidden_layers_arr = [2, 3, 4]
    #batch_size_arr = [32]
    #epochs_arr = [3]
    
    # Calculate total number of combinations
    n_combs = len(optimizer_arr) * len(reg_rate_l2_arr) * len(activation_arr) * \
        len(hidden_dim_arr) * len(hidden_layers_arr) * len(batch_size_arr) * len(epochs_arr)
    
    # Current index of combination
    ind_comb = 1
    
    res_gs_cv = []
    for optimizer in optimizer_arr:
        for reg_rate_l2 in reg_rate_l2_arr:
            for activation in activation_arr:
                for hidden_dim in hidden_dim_arr:
                    for hidden_layers in hidden_layers_arr:
                        for batch_size in batch_size_arr:
                            for epochs in epochs_arr:
                                
                                # Report info (verbose)
                                if verbose:
                                    print("GRID SEARCH:", ind_comb, "out of", n_combs)
                                ind_comb += 1
                                
                                # Get best result for given hyperparameters, cross-validated
                                res = cross_validate_nn(
                                    X_train_val,
                                    y_train_val,
                                    optimizer = optimizer,
                                    reg_rate_l2 = reg_rate_l2,
                                    activation = activation,
                                    hidden_units_in_layer = hidden_dim,
                                    hidden_layers = hidden_layers,
                                    batch_size = batch_size,
                                    epochs = epochs,
                                    verbose = True
                                )
                                res_gs_cv.append(res)
    
    # Choose the best result in grid search and cross-validation
    #res_best = None
    #acc_best = 0
    #for res in res_gs_cv:
    #    model = res['model']
    #    acc = res['val_accuracy']
    #    hist = res['history']
    #    if acc > acc_best:
    #        res_best = res
    
    # For each combination of hyperparameters, return dictionary from cross-val function
    return res_gs_cv

def train_nn_fixed(X_train_val, y_train_val):
    """
    Trains neural network with fixed architecture.

    Inputs:
    
    - X - input attributes, where every attribute has either float type or int type (for one-hot encoded categories);
    - y - output labels, where each label has integer type.

    Output: dictionary with the model and corresponding history (loss and accuracy score).
    """
    
    reg_l2_rate = 0.003
    hidden_dim = 50
    hidden_layers = 3
    activation = 'tanh'
    epochs = 50
    batch_size = 32
    val_split = 0.25

    # Prepare model
    model_nn = keras.models.Sequential()
    for i in range(hidden_layers):
        model_nn.add(
            layers.Dense(
                units = hidden_dim,
                activation = activation,
                kernel_regularizer = keras.regularizers.l2(reg_l2_rate),
                bias_regularizer = keras.regularizers.l2(reg_l2_rate),
                activity_regularizer = keras.regularizers.l2(reg_l2_rate)
            )
        )
    model_nn.add(
        layers.Dense(units=10)
    )
    model_nn.add(
        layers.Softmax()
    )

    # Compile model
    model_nn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    hist_nn = model_nn.fit(
        X_train_val,
        y_train_val,
        epochs = epochs,
        batch_size = batch_size,
        validation_split = val_split
    )

    # Prepare return dictionary
    res_dict = {
        'model' : model_nn,
        'training_history' : hist_nn
    }

    return res_dict