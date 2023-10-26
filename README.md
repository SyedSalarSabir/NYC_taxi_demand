End – 2 – End NYC Taxi demand predictor
Personal Project
● Downloaded the real historical data as parquet file from ‘https://www.nyc.gov/site/tlc/about/tlc-trip-recorddata.page’.
● Used pandas library for feature engineering. Converted the raw data to display number of rides per hour
between specified timeframes.
● For Model training used Mean Square Error to compare model i.e., baseline model, LightGBM and
XGBoost.
● Used to serverless service Hopsworks as feature Store. The trained model and historical data were stored on
this service.
● Feature pipeline and training pipeline were created using the Scikit Learn library.
● Machine Learning application was created using the StreamLit service.
