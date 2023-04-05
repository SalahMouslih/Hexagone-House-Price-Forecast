import pandas as pd
import numpy as np
from machine_learning.preprocess import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from machine_learning.preprocess import preprocess_ml
from machine_learning.preprocess import build_pipeline
from machine_learning.utilities import generate_feature_importance, save_result


QUARTILE = 0.05

def train_score_save(model, data, metropole, type_local):
    """
    Trains a machine learning model, computes its performance score, and saves the trained model to disk.
    
    Args:
        model (sklearn estimator): the machine learning model to be trained
        data (pandas dataframe): the input data
        metropole (str): the name of the metropolitan area
        type_local (str): the type of local (e.g., apartment, house)
        
    Returns:
        a pandas dataframe containing the performance metrics and characteristics of the input data
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Input data should be a pandas dataframe.')
    
    if not isinstance(metropole, str) or not isinstance(type_local, str):
        raise ValueError('Metropole and type_local should be strings.')
    
    try:
        data = train_test_split(data, metropole, type_local, split=False)
        data = preprocess_ml(data, type_local)
        shape = data.shape
        
        clf = build_pipeline(model, data)
        
        X_train, X_test, y_train, y_test = train_test_split(data, split=True, quartile=QUARTILE)
        clf.fit(X_train, y_train)
        
        best_score = clf.best_score_
        best_params = clf.best_params_
        
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        
        rmse = mean_squared_error(y_predict, y_test, squared=False)
        score = r2_score(y_predict, y_test)
        
        error = pd.DataFrame((np.abs(y_test - y_predict) / y_test) * 100)
        median = error.median()[0]
        mean = error.mean()[0]
        
        shape = data.shape
        estimator = clf.best_estimator_
        numerical_columns = list(data.select_dtypes(exclude=["object","string"]).columns)
        target='prix_m2_actualise'
        numerical_columns=[col for col in numerical_columns if col!=target]
        new_cat_cols = estimator.named_steps['preprocessor'].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(['code_departement'])
        ### Generate feature importance
        if model in ['xgboost','random_forest']:
            model_ = estimator[-1]
            generate_feature_importance(model_,model,metropole,type_local,numerical_columns,new_cat_cols)
        ### save to model to disk and print rsult in console

        result=save_result(estimator,type_local,model,metropole,best_score,best_params,rmse,score,median,mean,shape)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        result=pd.DataFrame()
    return result      