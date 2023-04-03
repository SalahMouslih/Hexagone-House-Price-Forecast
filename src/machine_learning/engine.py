import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestRegressor



def build_pipeline(model,data):
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
            params = {"poly_features__degree": [1,2],'model__alpha': np.arange(0, 0.2, 0.01)}
            poly_pipeline = Pipeline([('preprocessor',preprocessor),('poly_features', PolynomialFeatures()), ('model', LinearRegression())])
            clf = GridSearchCV(poly_pipeline, cv=folds, scoring='r2', param_grid=params,n_jobs=-1)
        elif model=='xgboost':
            params = {'xg__eta': [0.3, 0.02],'xg__n_estimators': [100,500,1000,5000]}
            xg = xgb.XGBRegressor(n_jobs = -1)
            xg_reg = Pipeline(
            steps=[("preprocessor", preprocessor), ("xg", xg)],verbose=True)
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