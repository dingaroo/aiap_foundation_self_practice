#!/opt/miniconda3/envs/aiap21_tech_asst/bin/python3

# Import Standard Python Library

# Third-party imports
import pandas as pd
import numpy as np
import yaml
from sklearn.utils._testing import ignore_warnings
from app_logging.app_logging import Logger

# Local applications/library specific imports
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

# Logging library
from app_logging.app_logging import Logger

# logging.basicConfig(level = logging.INFO)

# Environmental variables
config_path = './src/config.yaml'


@ignore_warnings(category = "Warning") # type: ignore
def main():
    # Loading configuration file
    try:
        with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
    except FileNotFoundError:
                Logger.error('Configuration file not found.')
                raise FileNotFoundError('Configuratrion file not found.')
    
    # Load csv file into a DataFrame
    try:
        df = pd.read_csv(config["file_path"])
    except FileNotFoundError:
        Logger.error('Data file not found.')
        raise FileNotFoundError('Data file not found.')
    except pd.errors.ParserError:
        Logger.error(f"\n❌ ERROR: The file at '{config["file_path"]}' is not a valid CSV.")
        print(f"\n❌ ERROR: The file at '{config["file_path"]}' is not a valid CSV.")
        print("Please check that the file is a standard, comma-separated text file.")
    
    # Initialize and run data preparation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)
    
    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(cleaned_df)
    
    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
            model_training.train_and_evaluate_baseline_models(
                    X_train, y_train, X_val, y_val
            )        
    )
    
    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
            X_train, y_train, X_val, y_val
    )
    
    # Combine all models and and their metrixs into dictionaies
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}
    
    # Find the best model based on R squared score
    best_model_name = max(all_metrics, key = lambda k: all_metrics[k]["R²"])
    best_model = all_models[best_model_name]
    logging.info(f'Best Model Found: {best_model_name}')
    
    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
            best_model, X_test, y_test, best_model_name
    )

   
if __name__ == "__main__":
    main()