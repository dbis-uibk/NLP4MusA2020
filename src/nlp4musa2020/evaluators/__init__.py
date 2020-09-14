"""Evaluators for nlp4musa2020."""
from sklearn.model_selection import KFold

def grid_parameters():
    """Grid parameter setup used to setup the grid search."""
    return {
        'verbose': 3,
        'cv': KFold(n_splits=5, shuffle=True),
        'refit': False,
        'scoring': ['neg_mean_absolute_error', 'neg_root_mean_squared_error'],
        'return_train_score': True,
    }

def grid_parameters_genres():
    """Grid parameter setup used to setup the grid search."""
    return {
        'verbose': 3,
        'cv': KFold(n_splits=5, shuffle=True),
        'refit': False,
        'scoring': ['f1_micro', 'f1_macro'],
        'return_train_score': True,
    }