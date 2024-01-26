from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def optimize_hyperparameters_grid(model, X_train, y_train, scoring_metric, param_range_factor=0.5, cv=5):
    """
    Description
    -----------
    Optimize hyperparameters for a given machine learning model using grid search.

    Parameters
    ----------    
    model :
        The machine learning model (classifier or regressor) for which hyperparameters need to be optimized.
    X_train : numpy array
              input matrix
              
    y_train : numpy array
              Target vector
              
    scoring_metric : sklearn metric 
        The metric used for model evaluation (e.g., 'accuracy', 'precision', 'recall', 'f1', 'mean_squared_error', etc.).

    param_range_factor : float
        The factor by which to vary the hyperparameter values around their defaults.

    cv : int
        Number of cross-validation folds.

    Returns
    -------
    new model
    
    """
    # Create a scorer based on the provided metric
    scorer = make_scorer(scoring_metric)

    # Define hyperparameter ranges based on the model's default values
    param_grid = {}

    for param_name, param_values in model.get_params().items():
        if isinstance(param_values, (int, float)):
            # If the parameter is numeric, create a range around the default value
            scale = max(1, abs(param_values) * param_range_factor)
            param_grid[param_name] = [param_values - scale, param_values, param_values + scale]

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring=scorer, cv=cv)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
