import pandas as pd
import warnings
from machine_learning.scores import train_score_save
from machine_learning.utils import read_data


warnings.simplefilter(action='ignore', category=FutureWarning)

def run_model(params,data):
    '''
    run machine learning model
    args:
    params :a tuple of  (metro,type_local,mod)
    data: preprocessed data
    '''
    (metro,type_local,mod)=params
    return train_score_save(data=data,model=mod,metropole=metro,type_local=type_local)
def ml_processing(path="data/processed/processed_data.csv"):
    '''
    Run machine learning model benchmark for all metropole,type_local, and model(Linear,xgboost,random)
    it will generate a result.txt,result.csv and feature importance for ensemble methods
    
    '''
    print("reading processed file")
    data=read_data(path)
    metropole=list(data.LIBEPCI.unique())
    type_bien=["Appartement",'Maison']
    model=['linear','xgboost','random_forest']
    # model=['xgboost']
    permutation=[(i,j,k) for i in metropole for j in type_bien for k in model ]
    results_dataframe=[]
    for p in permutation[:1]:
        print("Running model for ",p)
        result=run_model(p,data)
        results_dataframe.append(result)
    print("Save result dataframe to disk")    
    result=pd.concat(results_dataframe)
    result.to_csv('output/model/result.csv',index=False)


    

