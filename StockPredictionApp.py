import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import pandas_datareader as pdr
import keras
from sklearn.preprocessing import MinMaxScaler

START = '2015-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title('Stock Prediction app')

user_input = st.text_input("Enter Stock Ticker", 'PEP')
data_load_state = st.text("Loading data...")
data = pdr.DataReader(user_input, 'yahoo', START, TODAY)
data.reset_index(inplace=True)
data=data.sort_index(ascending=False,axis=0)
data_load_state.text("Loading data...Done!")

st.subheader("Details on last 1 week ")
st.write(data.head(7))

def plot_raw_data():
	layout = go.Layout(autosize=False, width=1000, height=700, xaxis= go.layout.XAxis(linecolor = 'black',linewidth = 1,
                          mirror = True), yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 1, mirror = True),
							margin=go.layout.Margin(l=50, r=50,b=100,t=100,pad = 4))
	fig = go.Figure(layout=layout)
	fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], close=data['Close'], high= data['High'],
								low=data['Low']))
	# fig.add_trace(go.Candlestick(x=data['Date'], =data['Close'], name ='Stock close'))
	fig.layout.update(title_text='Time chart', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

plot_raw_data()

data['MA100'] = data['Close'].rolling(100).mean()
data['MA200'] = data['Close'].rolling(200).mean()

def plot2():
	lay = go.Layout(autosize=False, width=1000, height=700, xaxis= go.layout.XAxis(linecolor = 'black',linewidth = 1,
	mirror = True), yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 1, mirror = True),
							margin=go.layout.Margin(l=50, r=50,b=100,t=100,pad = 4))
	fig2 = go.Figure(layout=lay)
	fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name ='Close price'))
	fig2.add_trace(go.Scatter(x=data['Date'], y=data['MA100'], name ='100 days moving avg'))
	fig2.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], name ='200 days moving avg'))
	fig2.layout.update(title_text='Closing price vs time chart with 100A & 200MA', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig2)

plot2()
 
# 3. Making predictions on the testing data
close_data = data.filter(['Close'])
dataset = close_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
training_data_len = math.ceil(len(dataset) * 0.7)
test_data = scaled_data[training_data_len - 60: , : ]
x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


# 	Loading model and predicting
model = keras.models.load_model('Stock_pred_model.h5')

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]
valid = data[training_data_len:]
 
valid['Predictions'] = predictions

def predictions():
	layout1 = go.Layout(autosize=False, width=1000, height=700, xaxis= go.layout.XAxis(linecolor = 'black',linewidth = 1,
                mirror = True), yaxis= go.layout.YAxis(linecolor = 'black', linewidth = 1, mirror = True),
                margin=go.layout.Margin(l=50, r=50,b=100,t=100,pad = 4))
	fig3 = go.Figure(layout=layout1)
	fig3.add_trace(go.Scatter(x=data['Date'], y=valid['Close'], name =' Original Close price'))
	fig3.add_trace(go.Scatter(x=data['Date'], y=valid['Predictions'], name ='Predictions'))
	fig3.layout.update(title_text='Predictions', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig3)


predictions()
