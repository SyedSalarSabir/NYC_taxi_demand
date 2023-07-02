from datetime import datetime , timedelta

import hopsworks
from hsfs.feature_store import FeatureStore

import pandas as pd
import numpy as np

import src.config as config
from src.paths import MODEL_DIR

def get_hopsworks_project()-> hopsworks.project.Project:
    project = hopsworks.login(
        project= config.HOPSWORKS_PROJECT,
        api_key_value= config.HOPSWORKS_API_KEY
    )
    return project


def get_feature_store()-> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features:pd.DataFrame)-> pd.DataFrame:

    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pu_location'] = features['pu_location'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def get_batch_data_from_feature_store(
        current_time : datetime
)-> pd.DataFrame:
    
    feature_store = get_feature_store()
    
    n_features = config.N_FEATURES

    fetch_data_from = current_time - timedelta(days=28)
    fetch_data_to = current_time - timedelta(hours=1)
    print(f'fetching data from {fetch_data_from} to {fetch_data_to}')

    
    feature_view = feature_store.get_feature_view(
        name = config.FEATURE_VIEW_NAME,
        version = config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days =1)),
        end_time= (fetch_data_to + timedelta(days = 1))
    )

    ts_data =  ts_data[ts_data['pu_hour'].between(fetch_data_from,fetch_data_to)]

    location_ids =  ts_data['pu_location'].unique()
    assert len(ts_data)==n_features * len(location_ids),\
     "Time-series data is not complete. Make sure your feature pipeline is up and runnning."
    
    x = np.ndarray(shape= (len(location_ids,n_features)),dtype=np.float32)

    for i, id in enumerate(location_ids):
        ts_data_i =  ts_data.loc[ts_data['pu_location']==id,:]
        ts_data_i = ts_data_i.sort_values(by = ['pu_hour'])
        x[i:] = ts_data_i['rides'].values

    features = pd.DataFrame(x,
                           columns = [f'rides_{x+1}_hr_before' for x in reversed(range(n_features))])    

    features['pu_hour'] = current_time
    features['pu_location'] = location_ids
    features.sort_by(by=['pu_location'],inplace = True)

    return features

def load_model_form_registry():
    import joblib
    from pathlib import Path

    project =  get_hopsworks_project()

    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name = config.MODEL_NAME,
        version = config.MODEL_VERSION
    )

    model_dir = model.download()
    model = joblib.load(Path(model_dir)/'model.pkl')
    return model


def load_model_from_computer():
    import joblib
    from pathlib import Path

    
    path_to_model = MODEL_DIR/"model_1_new.pkl"
    model= joblib.load(Path(path_to_model))

    return model