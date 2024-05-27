pip install prophet
import pandas as pd
import prophet
pip install fbprophet
pip install yfinance


#importing packages


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as spop

import warnings
warnings.filterwarnings('ignore')


ticker = 'TITAN.NS'
start = '2014-01-01'
end = '2024-01-01'

# Downloading data
df= yf.download(ticker, start, end,interval='1mo').reset_index()


df
df.to_csv('Titan.csv')


# Data Preparation:
# Load your time series data into a DataFrame

df = pd.read_csv('Titan.csv')

# Rename columns to 'ds' and 'y'
df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

pip install prophet

# Model Initialization and Fitting:
from prophet import Prophet

# Create a Prophet model
model = Prophet()

# Fit the model with your data
model.fit(df)

# Create a DataFrame with future dates for prediction
future = model.make_future_dataframe(periods=119)  # Adjust the number of periods as needed

# Generate predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)

# Plot components (trend, yearly seasonality, and weekly seasonality)
fig = model.plot_components(forecast)

