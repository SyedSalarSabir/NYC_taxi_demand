import pandas as pd
from typing import Tuple
def train_test_split(
        ts_data:pd.DataFrame,
        cut_off_year:int,
        cut_off_month:int,
        cut_off_day:int,
        target_column :str

) -> Tuple[pd.DataFrame,pd.Series,pd.DataFrame,pd.Series]:
    
    cut_of_date = f'{cut_off_year}-{cut_off_month:02d}-{cut_off_day:02d}'
    
    

    train = ts_data[ts_data['pu_hour'] < cut_of_date].reset_index(drop=True)
    test = ts_data[ts_data['pu_hour'] >= cut_of_date].reset_index(drop=True)
    
    X_train = train.drop(columns=[target_column])
    y_train = train[target_column]

    
    X_test = test.drop(columns=[target_column])
    y_test = test[target_column]

    return (X_train,X_test,y_train,y_test)
    