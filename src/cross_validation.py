from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

def cross_validate(model, X_train_val, y_train_val, k_value=4):
    """
    Perform k-fold cross validation on given data by training given model.

    This implementation uses fixed metrics and returns indices of data that were used for training and testing.
    """

    # Create fresh model with same hyperparameters
    model_curr = clone(model)

    # Output dictionary
    res_cv = {
        'estimator' : [],
        'test_accuracy' : [],
        'test_f1_macro' : [],
        'index_train' : [],
        'index_val' : []
    }

    # K-fold iterator
    kfold_iter = KFold(k_value)

    # Cross validation
    for train_index, val_index in kfold_iter.split(X_train_val):
        
        # Fresh model
        model_curr = clone(model)
        
        # Slice data
        X_train = X_train_val.iloc[train_index]
        y_train = y_train_val.iloc[train_index]
        X_val = X_train_val.iloc[val_index]
        y_val = y_train_val.iloc[val_index]
        
        # Train model on current fold
        model_curr.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model_curr.predict(X_val)
        acc_score = accuracy_score(y_val, y_pred)
        f1_macro_score = f1_score(y_val, y_pred, average='macro')
        
        # Save data
        res_cv['estimator'].append(model_curr)
        res_cv['test_accuracy'].append(acc_score)
        res_cv['test_f1_macro'].append(f1_macro_score)
        res_cv['index_train'].append(train_index)
        res_cv['index_val'].append(val_index)
    
    return res_cv