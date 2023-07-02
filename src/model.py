from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.pipeline import make_pipeline,Pipeline
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgbm



def average_4_week_rides (X:pd.DataFrame) -> np.array:
    X['average_rides_last_4_weeks'] = 0.25*(X[f'rides_{7*24}_hr_before']+\
                     X[f'rides_{14*24}_hr_before']+\
                     X[f'rides_{21*24}_hr_before']+\
                     X[f'rides_{28*24}_hr_before'])
    return X


class TemporalFeatureEngineering(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X,y=None):
        X_=X.copy()

        X_['hour'] = X_['pu_hour'].dt.hour
        X_['day_of_the_week'] = X_['pu_hour'].dt.dayofweek

        return X_.drop(columns = ['pu_hour'])
    
def training_pipeline(**hyperparameters)-> Pipeline:
    
    add_average_4_week_rides = FunctionTransformer(
    average_4_week_rides, validate=False
)
    

    add_temporal_data = TemporalFeatureEngineering()

  

    pipeline = make_pipeline(
    add_average_4_week_rides,
    add_temporal_data,
    lgbm.LGBMRegressor()
)
    return pipeline