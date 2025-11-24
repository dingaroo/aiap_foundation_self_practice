from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


def create_preprocessor(self) -> ColumnTransformer:
    """
    Creates a preprocessing pipeline for numerical, nominal and ordinal features

    Returns:
        ColumnTransformer: A preprocessor object that applies scaling, ecoding and imputation of the data
    """
    
    numerical_features = self.config['numerical_features']
    nominal_features = self.config['nominal_features']
    ordinal_features = self.config['ordinal_features']
    blood_pressure_categories = self.config['blood_pressure_categories']
    drinking_habits_categories = self.config['drinking_habits_categories']
    ordinal_mapping = [blood_pressure_categories, drinking_habits_categories]
    
    # Create pipelines for each type odf feature
    numerical_transformer = Pipeline(steps = [
        ('impute': SimpleImputer(strategy = 'median')),
        ('scale': StandardScaler())
    ])
    
    nominal_transformer = Pipeline(steps = [
        ('onehot': OneHotEncoder(handle_unknown = 'ignore'))
    ])
    
    ordinal_transformer = Pipeline(steps = [
        ('impute': SimpleImputer(strategy = 'most_frequent')),
        ('ordinal': OrdinalEncoder(categories = ordinal_mapping))
    ])
    
    # Combine all transformers into one preprocessor
    preprocessor = ColumnTransformer(transformers = [
        ('num', numerical_transformer, numerical_features),
        ('nom', nominal_transformer, nominal_features),
        ('ord', ordinal_transformer, ordinal_features)
    ], remainder = 'passthrough', n_jobs = 1)
    
    return preprocessor
