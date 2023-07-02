
from src.paths import RAW_DATA_DIR
import requests
import pandas as pd
import numpy as np
from typing import Optional , List
from tqdm import tqdm
from pathlib import Path
from numpy.distutils.misc_util import is_sequence


def download_data(year:int , month:int) -> Path:

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    #print('the url is ',url)
    response = requests.get(url)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'taxi_{year}-{month:02d}.parquet'
        open(path,'wb').write(response.content)
        return path
    else:
        raise Exception(f'{url} not available') 
    

def validation(
        dates_data : pd.DataFrame,
        year :int ,
                month: int) -> pd.DataFrame:
    
    starting_date = f'{year}-{month:02d}-01'
    ending_date = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    
   

    dates_data = dates_data[dates_data['pu_datetime'] >= starting_date]
    dates_data = dates_data[dates_data['pu_datetime'] <= ending_date]

    return dates_data

def load_raw_data(
        year : int,
        months : Optional[List[int]] = None
)-> pd.DataFrame:

    rides=  pd.DataFrame()

    if months is None:
        months = range(1,13)
    elif isinstance(months, int):
        months = [months]

    for month in months:
        
        FILE_PATH = RAW_DATA_DIR / f'taxi_{year}-{month:02d}.parquet'

        if not FILE_PATH.exists():
            try:
                print(f'downloading file taxi_{year}_{month:02d}.parquet')
                download_data(year=year,month=month)
            except:
                print(f'file taxi_{year}_{month:02d}.parquet not available')
                continue
        else:
            print(f'file taxi_{year}_{month:02d}.parquet already downloaded.')

        rides_one_month = pd.read_parquet(FILE_PATH)

        rides_one_month = rides_one_month[['tpep_pickup_datetime','PULocationID']]
    
        rides_one_month.rename(columns={'tpep_pickup_datetime':'pu_datetime',
                               'PULocationID':'pu_location'},inplace=True)
        
        rides_one_month = validation(rides_one_month, year=year, month=month)

        rides = pd.concat([rides,rides_one_month])

    rides = rides[['pu_datetime','pu_location']] 
    
    return rides


def add_missing_timeslots(ts_data: pd.DataFrame)->pd.DataFrame:
    
    location_ids = ts_data['pu_location'].unique()

    full_range = pd.date_range(ts_data['pu_hour'].min(), 
                               ts_data['pu_hour'].max(),
                                 freq ='H')
    
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):
           
            ts_data_i = ts_data.loc[ts_data['pu_location']==location_id, ['pu_hour','rides']]

            if ts_data_i.empty:
                ts_data_i = pd.DataFrame.from_dict(
                  {  'pu_hour' : ts_data['pu_hour'].max(), 'rides': 0 }
                )
            
            
            ts_data_i.set_index('pu_hour',inplace = True)

            ts_data_i.index = pd.DatetimeIndex(ts_data_i.index)
            
            ts_data_i = ts_data_i.reindex(full_range,fill_value=0)

            ts_data_i['pu_location'] = location_id
            
            output = pd.concat([output,ts_data_i])

    output = output.reset_index().rename(columns={'index':'pu_hour'})

    return output


def transform_to_ts_data(raw_data : pd.DataFrame) -> pd.DataFrame:
    
    raw_data['pu_hour'] = raw_data['pu_datetime'].dt.floor('H')


    agg_rides = raw_data.groupby(['pu_hour','pu_location']).size().reset_index()
    agg_rides.rename(columns={0:'rides'}, inplace = True)

    transformed_data = add_missing_timeslots(agg_rides)

    return transformed_data

def get_indicies(
        ts_data_i: pd.DataFrame,
        no_features :int,
        step_size: int
)-> pd.DataFrame:
    
    first_idx = 0
    middle_idx = no_features
    last_idx = middle_idx + 1
    stop_position = len(ts_data_i) - 1
    indices = []
    

    while last_idx <= stop_position:

        indices.append((first_idx,middle_idx,last_idx))
        first_idx +=step_size
        middle_idx +=step_size
        last_idx +=step_size  

    return indices  


def transform_ts_data_to_features_and_targets(
        ts_data :pd.DataFrame,
        no_features : int,
        step_size: int

 ) -> pd.DataFrame:

    features = pd.DataFrame()
    targets = pd.DataFrame()
    
    
    locations = ts_data['pu_location'].unique()

    for location in tqdm(locations):
        ts_data_i = ts_data.loc[ts_data['pu_location']==location, :].sort_values(by= ['pu_hour'])

        indices = get_indicies(ts_data_i=ts_data_i,
                               no_features=no_features,
                               step_size=step_size)
        
       

        no_examples = len(indices)
        X = np.ndarray(shape=(no_examples,no_features),dtype=np.float32)
        y = np.ndarray(shape=(no_examples),dtype=np.float32)
        pu_hours = []

       
        for i , idx in enumerate(indices):
            X[i, :] = ts_data_i.iloc[idx[0]:idx[1]]['rides'].values
            random = ts_data_i.iloc[idx[1]:idx[2]]['rides'].values
            

            y[i] = ts_data_i.iloc[idx[1]:idx[2]]['rides'].values
            pu_hours.append(ts_data_i.iloc[idx[1]]['pu_hour'])
        

        
        ts_data_one_location = pd.DataFrame(X,
                          columns=[f'rides_{x+1}_hr_before' for x in reversed(range(no_features))])
        ts_data_one_location['pu_location'] = location
        ts_data_one_location['pu_hour'] = pu_hours
        
        ts_target_one_location = pd.DataFrame(y, columns=['target_next_hour'])
        
        features = pd.concat([features,ts_data_one_location])
        targets = pd.concat([targets,ts_target_one_location])
    
    features.reset_index(drop=True, inplace=True)
    targets.reset_index(drop=True, inplace=True)

    return features, targets['target_next_hour']