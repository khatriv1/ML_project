import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    # Save to same artifacts folder as other components (src/components/artifacts/)
    trained_model_file_path: str = os.path.join(os.path.dirname(__file__), 'artifacts', "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test,
                models=models
            )

            # Print all model scores
            print(f"\n{'='*60}")
            print(f"{'MODEL PERFORMANCE REPORT':^60}")
            print(f"{'='*60}")
            print(f"{'Model Name':<30} {'R² Score':>15}")
            print(f"{'-'*60}")
            
            # Sort models by score (descending)
            sorted_models = sorted(model_report.items(), key=lambda x: x[1], reverse=True)
            
            for model_name, score in sorted_models:
                print(f"{model_name:<30} {score:>15.4f}")
            
            print(f"{'='*60}\n")

            # Get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")
            logging.info(f"Best model: {best_model_name} with R² score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Retrain best model on full training data
            best_model.fit(X_train, y_train)
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            
            print(f"{'='*60}")
            print(f"{'BEST MODEL SELECTED':^60}")
            print(f"{'='*60}")
            print(f"Model: {best_model_name}")
            print(f"Final R² Score: {r2_square:.4f}")
            print(f"{'='*60}\n")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)