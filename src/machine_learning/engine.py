''' Main engine for machine learning modeling'''

import pandas as pd
import warnings
from machine_learning.scores import train_score_save
from machine_learning.utilities import read_data
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_model(params, data):
    '''
    Run the machine learning model benchmark for all metropole, type_local, and model(Linear, xgboost, random).
    It will generate a result.txt, result.csv, and feature importance for ensemble methods.

    Args:
        params : tuple
            A tuple of (metro, type_local, mod)
        data : pandas.DataFrame
            Preprocessed data

    Returns:
        pandas.DataFrame
            Result of model training and testing
    '''
    try:
        (metro, type_local, mod) = params
        result = train_score_save(data=data, model=mod, metropole=metro, type_local=type_local)
        return result
    except Exception as e:
        print(f"An error occurred while running the model for {params}: {str(e)}")
        return None


def ml_engine(path="data/processed/processed_data.csv"):
    '''
    Run machine learning model benchmark for all metropole, type_local, and model(Linear, xgboost, random).
    It will generate a result.txt, result.csv, and feature importance for ensemble methods.

    Args:
        path : str, optional
            The path to the preprocessed data file

    Returns:
        None
    '''
    try:
        print("Reading processed file..")
        data = read_data(path)
        metropole = list(data.LIBEPCI.unique())
        type_bien = ["Appartement", 'Maison']
        model = ['linear', 'xgboost', 'random_forest']
        # model=['xgboost']
        permutation = [(i, j, k) for i in metropole for j in type_bien for k in model]
        results_dataframe = []
        for p in permutation:
            print("Running model for ", p)
            result = run_model(p, data)
            if result is not None:
                results_dataframe.append(result)
        print("Saving result dataframe to disk..")
        result = pd.concat(results_dataframe)
        result.to_csv('output/model/results.csv', index=False)
    except Exception as e:
        print(f"An error occurred while processing the machine learning model: {str(e)}")   
