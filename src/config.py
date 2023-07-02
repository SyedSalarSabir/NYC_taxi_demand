from dotenv import load_dotenv
from src.paths import PATRENT_DIR
import os

#HOPSWORKS_PROJECT = 'my_taxi_demand'
HOPSWORKS_PROJECT = 'taxi_demand_salar'


try:
    load_dotenv(PATRENT_DIR/'.env')
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception ('Create a .env file with HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'ts_data_feature_group'
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = 'ts_data_feature_view'
FEATURE_VIEW_VERSION = 1

MODEL_NAME= 'model_to_predict_taxi_demand_for_the_next_hour'
MODEL_VERSION= 1

N_FEATURES = 24 * 28

STEP_SIZE = 24