import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

def save_result(estimator,type_local,model,metropole,best_score,best_params,rmse,score,median,mean,shape):
            '''
            Save result as txt file and csv file to disk and dump model
            Args:
            estimator: a sklearn model
            type_local: str
            metropole: str
            best_score:float
            best_parms: dict
            rsme: float
            score:  float
            median: float
            mean: float
            shape:tuple

            Return dataframe
            '''
            result = f"{metropole}-{type_local}-{model}-best_score: {best_score}, best_param: {best_params}, rmse: {rmse}, score: {score}, median: {median}, mean: {mean}"
            print(result)
            print('saved model to disk')
            dump(estimator, f"output/model/results_dumps/{metropole.strip().replace(' ', '-')}-{type_local}-{model}.joblib")
        

            dataframe = pd.DataFrame([[metropole, type_local, model, best_score, best_params, rmse, score, median, mean, shape]],
                                 columns=['metropole', 'type_local', 'model', 'best_score_cv_search', 'best_params', 'rmse', 'score_r2_test', 'error_prix_actualise_median', 'error_prix_actualise_mean', 'shape'])
            with open("output/model/results.txt", "a+") as f:
                f.write(result + '\n')
            return dataframe
            
def read_data(path):
        try: 
            df = pd.read_csv(path)
        except IOError as e:
             print(e)
             print('make sure you have generated the processed file first')
        return df

def generate_feature_importance(model_,model,metropole,type_local,numerical_columns,new_cat_cols):
        '''
        Generate feature importance graph and save to disk
        '''
        importance=list(zip(numerical_columns+list(new_cat_cols), model_.feature_importances_))
        df_importances = pd.DataFrame(importance,columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
        df_importances.Importance = (df_importances.Importance / sum(df_importances.Importance)) * 100
        
        #plot feature importance
        plt.figure(figsize=(8,20))
        plt.barh(data=df_importances,y='Feature', width='Importance',color='#ff9600')
        y=list(df_importances.Importance)
        for i in range(len(y)):
            plt.text(x= round(y[i],2),y= i,s= round(y[i],2), c='b')
        plt.xlabel('feature_importance(%)')
        plt.ylabel('features')
        plt.title('Analysis of feature importance for model:{}-{}-{}'.format(model,metropole,type_local))
        
        #save metrics
        name='output/model/Feature_importance/{}-{}-{}.png'.format(model,metropole,type_local)
        plt.savefig(name, bbox_inches='tight')
        print('feature importance graph save to:',name)
        