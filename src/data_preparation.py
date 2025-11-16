# # Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess students' examination results data.

    Attributes:
    -----------
    config : Dict[str, Any] - Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer - A preprocessor pipeline for transforming numerical and nominal features.
    """
    

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()


    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")
        df = df.drop_duplicates(inplace=True)
        # df["flat_type"] = df["flat_type"].replace("FOUR ROOM", "4 ROOM", inplace=True)
        # df["lease_commence_date"] = df["lease_commence_date"].abs()
        # df["storey_range"] = df["storey_range"].apply(self._convert_storey_range)
        # df = self._fill_missing_names(df, "town_id", "town_name")
        # df = self._fill_missing_names(df, "flatm_id", "flatm_name")
        # df = df.drop(columns=["id", "town_id", "flatm_id"])
        # df["year_month"] = pd.to_datetime(df["month"], format="%Y-%m")
        # df["year"] = df["year_month"].dt.year
        # df["month"] = df["year_month"].dt.month
        # df = df.drop(columns=["year_month"])
        # df["remaining_lease_months"] = df["remaining_lease"].apply(
        #     self._extract_lease_info
        # )
        # df = df.drop(columns=["remaining_lease", "block", "street_name"])
        

        # # Converting all values to uppercase
        # df['CCA'] = df['CCA'].str.upper()

        # # Filling the missing values with the 'NONE' category
        # df['CCA'] = df['CCA'].fillna('NONE')
        df = self._clean_CCA(df, 'CCA')

        
        # Convert the 2 string columns to datetime objects
        df['sleep_time_dt'] = pd.to_datetime(df['sleep_time'], format='%H:%M')
        df['wake_time_dt'] = pd.to_datetime(df['wake_time'], format='%H:%M')

        # Calculate the duration, not forgetting to handle the overnight timings
        duration = np.where(
            df['wake_time_dt'] < df['sleep_time_dt'],
            # If wake time is "before" sleep time, add a day to wake time
            df['wake_time_dt'] + pd.Timedelta(days=1) - df['sleep_time_dt'],
            # Otherwise, it's a simple subtraction
            df['wake_time_dt'] - df['sleep_time_dt']
        )
        df['sleep_duration'] = duration

        # Convert the duration (Timedelta) to total minutes
        df['sleep_minutes'] = (df['sleep_duration'].dt.total_seconds() / 60).astype(int)
        
        # Create replacement mapping values
        tuition_replacement_code = {'Yes': 'Y', 
                                    'No': 'N'
                                    }
        # Perform the replacement operationg
        df['tuition'] = df['tuition'].replace(tuition_replacement_code)

        # Calculate the median value of the attendance_rate column
        attendance_rate_median = df['attendance_rate'].median()

        # Assign the missing values with the calculated median 
        df['attendance_rate'] = df['attendance_rate'].fillna(attendance_rate_median)
        
        # Setting absolute values for the 'age' column
        df['age'] = df['age'].abs()

        # Create 'age' replacement mapping values
        age_replacement_map = {4: 15,
                            5: 15,
                            6: 16
                            }

        # Perform the replacement operationg
        df['age'] = df['age'].replace(age_replacement_map)
        
        # Removing non-essential colukns
        df.drop(columns = ['index', 'student_id', 'sleep_time', 'wake_time', 'sleep_time_dt', 
                           'wake_time_dt', 'sleep_duration'],
                inplace = True)
        
        logging.info("Data cleaning completed.")
        return df
        
        
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, nominal, and ordinal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor


    @staticmethod
    def _clean_CCA(dataset: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Cleans up "CCA" column in the DataFrame by converting all values to uppercase 
        and filling missing values with "NONE".

        Args:
            df (pd.DataFrame): DataFrame containing the "CCA" column.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        # Converting all values to uppercase
        dataset[col] = dataset[col].str.upper()

        # Filling the missing values with the 'NONE' category
        dataset[col] = dataset[col].fillna('NONE')
        
        return dataset
        
    
    # @staticmethod
    # def _calculate_sleep_time(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Convert 'sleep_time' and 'wake_time' columns into datetime format. 
    #     Calculates sleeping time by obtaining the difference between sleep_time and wake_time. 
    #     Then converts the sleep duration into minutes.

    #     Args:
    #         df (pd.DataFrame): DataFrame containing the 'sleep_time' and 'wake_time' columns.

    #     Returns:
    #         pd.DataFrame: Updated DataFrame with the 'sleep_minutes' column.
    #     """
    #     # Convert the 2 string columns to datetime objects
    #     df['sleep_time_dt'] = pd.to_datetime(df['sleep_time'], format='%H:%M')
    #     df['wake_time_dt'] = pd.to_datetime(df['wake_time'], format='%H:%M')

    #     # Calculate the duration, not forgetting to handle the overnight timings
    #     duration = np.where(
    #         df['wake_time_dt'] < df['sleep_time_dt'],
    #         # If wake time is "before" sleep time, add a day to wake time
    #         df['wake_time_dt'] + pd.Timedelta(days=1) - df['sleep_time_dt'],
    #         # Otherwise, it's a simple subtraction
    #         df['wake_time_dt'] - df['sleep_time_dt']
    #     )
    #     df['sleep_duration'] = duration

    #     # Convert the duration (Timedelta) to total minutes
    #     df['sleep_minutes'] = (df['sleep_duration'].dt.total_seconds() / 60).astype(int)
        
    #     return df
    
    
    # @staticmethod
    # def _normalize_tuition(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Standardize entries in the 'tuition' column to only 'Y' and 'N' values. 

    #     Args:
    #         df (pd.DataFrame): Dataframe containing the 'tuition' column.

    #     Returns:
    #         pd.DataFrame: The updated DataFrame with the 'tuition' column normalized.
    #     """
    #     # Create replacement mapping values
    #     tuition_replacement_code = {'Yes': 'Y', 
    #                                 'No': 'N'
    #                                 }
    #     # Perform the replacement operationg
    #     df['tuition'] = df['tuition'].replace(tuition_replacement_code)
        
    #     return df
    
    
    # @staticmethod
    # def _normalize_age(df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Normalize the age to 15, 16
    #     Fill missing values with the median values of the students' ages

    #     Args:
    #         df (pd.DataFrame): DataFrame containing the 'age' column. 

    #     Returns:
    #         pd.DataFrame: The normalized DataFrame. 
    #     """
    #     # Calculate the median value of the attendance_rate column
    #     attendance_rate_median = df['attendance_rate'].median()

    #     # Assign the missing values with the calculated median 
    #     df['attendance_rate'] = df['attendance_rate'].fillna(attendance_rate_median)
        
    #     return df
    

    # @staticmethod
    # def _convert_storey_range(storey_range: str) -> float:
    #     """
    #     Converts a storey range string into its average numerical value.

    #     Args:
    #     -----
    #     storey_range (str): A string representing a range of storeys, in the format 'XX TO YY'.

    #     Returns:
    #     --------
    #     float: The average value of the two storeys in the range.
    #     """
    #     range_values = storey_range.split(" TO ")
    #     return (int(range_values[0]) + int(range_values[1])) / 2



    # @staticmethod
    # def _fill_missing_names(df: pd.DataFrame, id_column: str, name_column: str) -> pd.DataFrame:
    #     """
    #     Fills missing values in the 'name_column' using the 'id_column'.

    #     Args:
    #     -----
    #     df (pd.DataFrame): The DataFrame containing the columns to be filled.
    #     id_column (str): The name of the column containing the IDs.
    #     name_column (str): The name of the column containing the names to be filled.

    #     Returns:
    #     --------
    #     pd.DataFrame: The DataFrame with missing values in 'name_column' filled.
    #     """
    #     missing_names = df[name_column].isna()
    #     name_mapping = (
    #         df[[id_column, name_column]]
    #         .dropna()
    #         .drop_duplicates()
    #         .set_index(id_column)[name_column]
    #         .to_dict()
    #     )
    #     df.loc[missing_names, name_column] = df.loc[missing_names, id_column].map(
    #         name_mapping
    #     )
    #     return df


    # @staticmethod
    # def _extract_lease_info(lease_str: str) -> int:
    #     """
    #     Converts lease information from a string format to total months.

    #     Args:
    #     -----
    #     lease_str (str): The remaining lease period as a string.

    #     Returns:
    #     --------
    #     int: The total number of months, or None if the input is NaN.
    #     """
    #     if pd.isna(lease_str):
    #         return None
    #     years_match = re.search(r"(\d+)\s*years?", lease_str)
    #     months_match = re.search(r"(\d+)\s*months?", lease_str)
    #     number_match = re.match(r"^\d+$", lease_str.strip())
    #     if years_match:
    #         years = int(years_match.group(1))
    #     elif number_match:
    #         years = int(number_match.group(0))
    #     else:
    #         years = 0
    #     months = int(months_match.group(1)) if months_match else 0
    #     total_months = years * 12 + months
    #     return total_months
    
    
    
    #### END OF SCRIPT ####
