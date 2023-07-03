import streamlit as st
from datetime import datetime , timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pydeck as pdk
import requests
import zipfile

from src.plot import plot_one_example
from src.paths import DATA_DIR
from src.inference import (
    load_model_form_registry,
    load_model_from_computer,
    get_batch_data_from_feature_store,
    get_model_predictions
)
from src.data import(load_raw_data , 
                     transform_to_ts_data,
                     )
from src.load_data_from_computer import load_features_from_computer 


st.set_page_config(layout='wide')


st.title('Taxi Demand Predictor')

time_now = pd.to_datetime(datetime.utcnow()).floor('H')

st.header(f'Time Now {time_now}')


progress_bar =  st.sidebar.header('Work in progress')
progress_bar = st.sidebar.progress(0)

N_steps = 7

@st.cache_data
def load_shape_data():
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / 'taxi_zones/taxi_zone.zip'
    if response.status_code ==200:
        open(path,'wb').write(response.content)
    else:
        print(f'The {URL} is not available')
    
    with zipfile.ZipFile(path,'r') as zip:
        zip.extractall(DATA_DIR/'taxi_zones')
    
    return gpd.read_file(DATA_DIR/'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')


with st.spinner(text= 'Downloading shape file for plotting taxi zones'):
     gpd_pd = load_shape_data()
     st.sidebar.write('The shape file is downloaded')
     progress_bar.progress(1/N_steps)


# with st.spinner(text ='Fetching the most inference data'):
#     features =  get_batch_data_from_feature_store(current_time=time_now)
#     st.spinner.write('Inference data retrived')
#     progress_bar.progress(2/N_steps)
#     print(f'{features}')

@st.cache_resource
def lf(time:datetime):
    return load_features_from_computer(current_time= time)

with st.spinner(text ='Fetching the inference data'):
    features =  lf(time= time_now)
    st.sidebar.write('Inference data retrived')
    progress_bar.progress(2/N_steps)
    print(f'{features}')

with st.spinner(text='Loading model from Model Registry'):
    model = load_model_from_computer()
    st.sidebar.write('Model loaded from Registry')
    progress_bar.progress(3/N_steps)

with st.spinner(text='Loading Prediction from Model Registry'):
    results = get_model_predictions(model, features)
    st.sidebar.write('Predictions loaded from Registry')
    progress_bar.progress(4/N_steps)



with st.spinner(text='Preparing data for plotting'):

    def color_scaling(x,minval,maxval,start_color,end_color):

        f = float(x-minval)/(maxval-minval)
        scale = tuple(f*(b-a)+a for (a, b) in zip(start_color, end_color))
        return scale
    df = pd.merge(gpd_pd, results, 
    right_on = 'pu_location',left_on = 'LocationID', how ='inner')

    BLACK, GREEN = (0,0,0), (0,255,0)

    df['color_scale'] = df['predicted_demand']
    max_value, min_value = df['color_scale'].max(), df['color_scale'].min()
    
    @st.cache_data
    def apply_color():
        return df['color_scale'].apply(lambda x : color_scaling(x,min_value,
                                                                max_value,BLACK,GREEN))
   
    df['color'] = apply_color()
    st.sidebar.write('Data prepared for plotting')
  
    progress_bar.progress(5/N_steps)


with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    st.sidebar.write('NYC map generated')

    progress_bar.progress(6/N_steps)

with st.spinner(text="Plotting time-series data"):
   
    row_indices = np.argsort(results['predicted_demand'].values)[::-1]
    n_to_plot = 10

   
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_example(
            features=features,
            targets=results['predicted_demand'],
            example_id=row_id,
            prediction=pd.Series(results['predicted_demand'])
        )
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, width=1000)

    progress_bar.progress(7/N_steps)
