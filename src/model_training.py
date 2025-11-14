# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipelineds
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelTraining:
    """
    A class used to train and evaluate different regression models.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for model training.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
        """
        Initializes the ModelTraining class with a configuration dictionary and a preprocessor.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for model training.
        preprocessor (ColumnTransformer): Fitted preprocessor from DataPreparation.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
        }

    def train_model(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Trains and evaluates regression models.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing features and target variable.

        Returns:
        --------
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            A tuple containing:
            - best_models (Dict[str, Any]): Dictionary of the best trained models.
            - best_params (Dict[str, Any]): Dictionary of the best parameters found for each model.
            - model_performance (Dict[str, Any]): Dictionary of performance metrics for each model.
        """
        logging.info("Starting model training...")

        # Split data into features and target
        X = df.drop(columns=self.config["target_column"])
        y = df[self.config["target_column"]]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
        )
