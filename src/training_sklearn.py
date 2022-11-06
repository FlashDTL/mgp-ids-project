from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

def train_rf_grid_cv(X, y, param_grid_rf):
    """
    Trains random forest models using exhaustive grid search and cross-validation.

    Inputs: X, y.

    Output: dictionary with best estimator and corresponding score (accuracy).
    """

    # Prepare grid search cross-val model and a grid of parameters
    model_rf = RandomForestClassifier()
    grid_search_cv_rf = GridSearchCV(
        estimator=model_rf,
        param_grid=param_grid_rf,
        scoring='accuracy',
        cv=4,
        verbose=2
    )

    # Train models
    res_gscv = grid_search_cv_rf.fit(X, y)

    # Return best estimator
    res = {
        'best_estimator' : grid_search_cv_rf.best_estimator_,
        'best_accuracy' : grid_search_cv_rf.best_score_,
        'best_params' : grid_search_cv_rf.best_params_
    }

    return res