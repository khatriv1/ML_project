import sys
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


def get_hyperparameter_grids():
    """
    Returns hyperparameter grids for all models
    """
    try:
        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "Decision Tree": {
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "Ridge Regression": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "Lasso Regression": {
                "alpha": [0.1, 1.0, 10.0, 100.0]
            },
            "K-Neighbors Regressor": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            },
            "XGBRegressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "CatBoosting Regressor": {
                "iterations": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "depth": [4, 6, 8]
            },
            "AdaBoost Regressor": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            }
        }
        return param_grids
    except Exception as e:
        raise CustomException(e, sys)


def get_base_models():
    """
    Returns base models without hyperparameters
    """
    try:
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }
        return models
    except Exception as e:
        raise CustomException(e, sys)


def tune_hyperparameters(X_train, y_train, models, param_grids, cv=5, scoring='r2'):
    """
    Performs hyperparameter tuning for all models using GridSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        models: Dictionary of model names and model instances
        param_grids: Dictionary of parameter grids for each model
        cv: Number of cross-validation folds (default: 5)
        scoring: Scoring metric (default: 'r2')
    
    Returns:
        Dictionary of best models with tuned hyperparameters
    """
    try:
        tuned_models = {}
        
        logging.info("Starting hyperparameter tuning for all models...")
        
        for model_name, model in models.items():
            if model_name not in param_grids:
                # If no hyperparameters defined, use base model
                logging.info(f"No hyperparameters defined for {model_name}, using base model")
                tuned_models[model_name] = model
                continue
            
            logging.info(f"Tuning hyperparameters for {model_name}...")
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=cv,
                scoring=scoring,
                n_jobs=-1,  # Use all available cores
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_score = grid_search.best_score_
            
            logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logging.info(f"Best CV score for {model_name}: {best_score:.4f}")
            
            tuned_models[model_name] = best_model
        
        logging.info("Hyperparameter tuning completed for all models")
        return tuned_models
        
    except Exception as e:
        raise CustomException(e, sys)

