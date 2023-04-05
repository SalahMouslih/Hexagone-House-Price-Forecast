import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

trimestres=['2021-T3', '2021-T4', '2022-T1', '2022-T2']
to_drop=['adresse_numero', 'adresse_suffixe','numero_disposition',
       'adresse_code_voie', 'code_postal', 'code_commune',
       'ancien_code_commune','ancien_nom_commune' , 'ancien_id_parcelle',
       'numero_volume', 'lot1_numero', 'lot1_surface_carrez', 'lot2_numero',
       'lot2_surface_carrez', 'lot3_numero', 'lot3_surface_carrez',
       'lot4_numero', 'lot4_surface_carrez', 'lot5_numero',
       'lot5_surface_carrez','code_type_local',
       'code_nature_culture', 'nature_culture', 'code_nature_culture_speciale',
       'nature_culture_speciale','id_mutation','id_parcelle','numero_disposition', 'nature_mutation','valeur_fonciere',
       'id_parcelle','nature_mutation','date_mutation','LIBEPCI','prix_actualise','DCOMIRIS','DCIRIS','prix_m2',
       'type_local','geometry','indices','quantile_prix','coeff_actu']

def train_test_split(df, metropole=None, type_local=None, random_state=42, split=True, trimestres=trimestres, quartile=None):
    """
    Splits a given dataframe into training and testing sets based on the given parameters.

    Args:
        df (pandas.DataFrame): The dataframe to split.
        metropole (str): The metropole to filter by.
        type_local (str): The type of property to filter by.
        random_state (int): The random state to use for shuffling the data.
        split (bool): Whether to split the data into training and testing sets or not.
        trimestres (list): The trimesters to exclude from the training set.
        quartile (float): The quartile to use for filtering by price.

    Returns:
        tuple or pandas.DataFrame: If `split` is True, returns a tuple of (train_x, test_x, train_y, test_y). Otherwise, returns a pandas.DataFrame of the full split dataset.
    """
    # Check if quartile is a valid number between 0 and 1
    if quartile is not None and not (0 <= quartile <= 1):
        raise ValueError("Quartile must be a number between 0 and 1.")

    # Filter by price quartile if quartile is specified
    if quartile:
        df = df[df.prix_m2_actualise < df.prix_m2_actualise.quantile(1 - quartile)]
        df= df[df.prix_m2_actualise > df.prix_m2_actualise.quantile(quartile)]

    # Filter by metropole if specified
    if metropole:
        df = df[df['LIBEPCI'].str.contains(metropole)]

    # Filter by type_local if specified
    if type_local:
        df = df[df['type_local'].str.contains(type_local)]

    # Shuffle entire dataframe
    shuffled_df = df.sample(frac=1, random_state=random_state)

    # Split into training and testing sets if specified
    if split:
        # Select training data
        train_df = shuffled_df[~shuffled_df['trimestre_vente'].isin(trimestres)]
        train_x = train_df.drop('prix_m2_actualise', axis=1)
        train_y = train_df['prix_m2_actualise']

        # Select test data
        test_df = shuffled_df.drop(train_df.index)
        test_x = test_df.drop('prix_m2_actualise', axis=1)
        test_y = test_df['prix_m2_actualise']

        return train_x, test_x, train_y, test_y
    else:
        # Include test data in the full split dataset
        test_df = shuffled_df.copy()
        test_df['prix_m2_actualise'] = test_df['prix_m2']
        return pd.concat([shuffled_df, test_df], axis=0)

def preprocess_ml(data, type_local):
    """
    Clean the input data for use in machine learning models.

    Args:
        data (pd.DataFrame): The input dataframe to be cleaned.
        type_local (str): The type of local to be cleaned (e.g. 'Appartement', 'Maison').

    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    
    # Drop unnecessary columns
    drop_clean=list(set(data.columns)&set(to_drop))
    data  = data.drop(drop_clean, axis=1)
    
    # get numerical columns and calculate correlation matrix
    numerical_columns = list(data.select_dtypes(exclude=["object","string"]).columns)
    target='prix_m2_actualise'
    numerical_columns=[col for col in numerical_columns if col!=target]
    corr_matrix = data[numerical_columns].corr().abs()

    # Remove highly correlated columns
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop_col = [column for column in upper.columns if any(upper[column] > 0.95)]
    data = data.drop(to_drop_col, axis=1)

    # Drop surface_terrain column for apartments
    if type_local=="Appartement":
        data = data.drop(['surface_terrain'], axis=1)
        
    # Drop rows with missing values
    data = data.dropna()

    return data


def build_pipeline(model,data):
    '''
    Build a pipeline using RandomSearchCV or GridSearchCV to optimize the hyperparameters for the specified machine learning model
    Args:
    model : str, the name of the machine learning model to be used
    data : pandas DataFrame, preprocessed data to be used in the pipeline
    Returns:
    clf : the resulting pipeline after applying the RandomSearchCV or GridSearchCV
    '''
    numerical_columns = list(data.select_dtypes(exclude=["object","string"]).columns)
    target='prix_m2_actualise'
    numerical_columns=[col for col in numerical_columns if col!=target]
    categorical_columns=['code_departement'] 
    folds = 3
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
        )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,numerical_columns),
            ("cat", categorical_transformer,categorical_columns),
        ])
    rd=RandomForestRegressor(n_jobs=-1)
    xg_reg = xgb.XGBRegressor(n_jobs=-1)
    if model=='linear':
        params = {"poly_features__degree": [1,2]}
        poly_pipeline = Pipeline([('preprocessor',preprocessor),('poly_features', PolynomialFeatures()), ('model', LinearRegression())])
        clf = GridSearchCV(poly_pipeline, cv=folds, scoring='r2', param_grid=params,n_jobs=-1)
    elif model=='xgboost':
        params = {'xg__eta': [0.3, 0.02],'xg__n_estimators': [100,500,1000,5000]}
        xg = xgb.XGBRegressor(n_jobs = -1)
        xg_reg = Pipeline(
        steps=[("preprocessor", preprocessor), ("xg", xg)],verbose=False)
        clf = RandomizedSearchCV(xg_reg, param_distributions=params, n_iter=2,
        scoring='r2', n_jobs=-1, cv=folds,random_state=1001 )
        
    else:
        rd_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("rd", rd)]
        )
        params = {
            'rd__n_estimators':[100,150,200,300]
        }
        clf= RandomizedSearchCV(rd_pipeline, param_distributions=params, n_iter=2,
                scoring='r2', n_jobs=-1, cv=folds,
                                random_state=1001 )            
    return clf