from datetime import datetime, timedelta
from src.data import load_raw_data , transform_to_ts_data, transform_ts_data_to_features_and_targets
import pandas as pd
import src.config as config
import numpy as np




def fetch_data(
        data_from : datetime,
        data_to : datetime
)-> pd.DataFrame:

    fetch_data_from_ = data_from - timedelta(days=7*52)
    fetch_data_to_ = data_to - timedelta(days=7*52)

    rides = load_raw_data(fetch_data_from_.year,fetch_data_from_.month)
    rides = rides[rides['pu_datetime'] >= fetch_data_from_]
    rides2 = load_raw_data(fetch_data_to_.year,fetch_data_to_.month)
    rides2 = rides2[rides2['pu_datetime']<fetch_data_to_]

    rides = pd.concat([rides,rides2])
    rides['pu_datetime'] += timedelta(days=7*52)

    rides.sort_values(by=['pu_location','pu_datetime'],inplace= True)

    return rides

def load_features_from_computer(current_time:datetime)->pd.DataFrame:
    
    current_time = pd.to_datetime(datetime.utcnow()).floor('H')
    print(f'current time {current_time}')

    fetch_data_from = current_time- timedelta(days=28)
    fetch_data_to = current_time
    
    historical_data = fetch_data(data_from=fetch_data_from,
                                 data_to=fetch_data_to)
    
    transformed_data = transform_to_ts_data(historical_data)

    n_features = config.N_FEATURES

    fetch_data_from = current_time - timedelta(days=28)
    fetch_data_to = current_time - timedelta(hours=1)
    print(f'fetching data from {fetch_data_from} to {fetch_data_to}')

    transformed_data =  transformed_data[transformed_data['pu_hour'].between(fetch_data_from,fetch_data_to)]
    
    location_ids =  transformed_data['pu_location'].unique()
    assert len(transformed_data)==n_features * len(location_ids),\
     "Time-series data is not complete. Make sure your feature pipeline is up and runnning."
    
    x = np.ndarray(shape= (len(location_ids),n_features),dtype=np.float32)

    for i, id in enumerate(location_ids):
        transformed_data_i =  transformed_data.loc[transformed_data['pu_location']==id,:]
        transformed_data_i = transformed_data_i.sort_values(by = ['pu_hour'])
        x[i:] = transformed_data_i['rides'].values

    features = pd.DataFrame(x,
                           columns = [f'rides_{x+1}_hr_before' for x in reversed(range(n_features))])    

    features['pu_hour'] = current_time
    features['pu_location'] = location_ids
    

    return features

    