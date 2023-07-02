from typing import Optional, List
from datetime import timedelta
import pandas as pd
import plotly.express as px

def plot_one_example(
        features : pd.DataFrame,
        targets : pd.Series,
        example_id : int,
        prediction: Optional[pd.Series] = None
):
    features_ = features.iloc[example_id]
    
    targets_ = targets.iloc[example_id]
  
    ts_columns = [c for c in features.columns if c.startswith('rides_')]
    ts_values = [features_[c] for c in ts_columns] + [targets_]
    ts_dates = pd.date_range(
        features_['pu_hour'] - timedelta(hours=len(ts_columns)),
        features_['pu_hour'], freq='H'
    )


    title = f"Location ID : {features_['pu_location']} , Pick Up hour: {features_['pu_hour']}"

    fig = px.line(
        x = ts_dates, y = ts_values,
        title=title, markers= True,template= 'plotly_dark'
    )

    fig.add_scatter(
        x = ts_dates[-1:], y = [targets_], line_color= 'green', mode = 'markers', marker_size = 10, 
        name='Actual demand'
    )

    if prediction is not None:
          prediction_ = prediction.iloc[example_id]
          fig.add_scatter(
        x = ts_dates[-1:], y = [prediction_], line_color= 'red', mode = 'markers', marker_size = 10, 
        name='Predicted demand', marker_symbol='x'
    )
    return fig
