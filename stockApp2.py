import datetime
from numpy import array
import streamlit as st
from datetime import date
import yfinance as yf
# from fbprophet import Prophet
# from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from plotly import graph_objs as go
import math

START = "2010-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Open'],
                             marker={'color': 'red'},
                             name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'],
                  y=data['Close'],
        marker={'color': 'blue'},
        name="stock_close"))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # fig.update_traces(line_color='purple')
    st.plotly_chart(fig)


plot_raw_data()
# create a new dataframe with only the close column
aapl = yf.Ticker(selected_stock)
date = datetime.datetime.now()
date = date.strftime("%Y-%m-%d")
df = aapl.history(start="2010-01-01", end=date)
data = df.filter(['Close'])
# convert the dataframe to numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)*.8)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
scaled_data = np.array(scaled_data)
scaled_data = np.reshape(scaled_data, (len(scaled_data), 1))
train_data = scaled_data[0:training_data_len, :]

best_window_size = 63
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(best_window_size, len(train_data)):
    x_train.append(train_data[i-best_window_size:i, 0])
    y_train.append(train_data[i, 0])
# Create the testing data set
# Create a new array containing scaled values from index 2101 to 2701
test_data = scaled_data[training_data_len-best_window_size:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
y_test_dummy = []
for i in range(best_window_size, len(test_data)):
    x_test.append(test_data[i-best_window_size:i, 0])
    y_test_dummy.append(test_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# Convert the data to a numpy array
x_test = np.array(x_test)
# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

model = load_model('m.h5')

model.fit(x_train, y_train, epochs=20, verbose=0)

x_input = scaled_data[len(data)-best_window_size:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for specified period

lst_output = []
n_steps = best_window_size
i = 0
while (i < period):

    if (len(temp_input) > best_window_size):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        # print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1


# print(lst_output)
lst_output = np.array(lst_output)
lst_output = scaler.inverse_transform(lst_output)

shreyas = []
date = datetime.datetime.now()
for i in range(period):
    shreyas.append(date)
    print(date.date())
    date += datetime.timedelta(days=1)
    if (date.weekday() == 5):
        date += datetime.timedelta(days=2)
# import pandas library as pd

# create an Empty DataFrame object With
# column names and indices
df = pd.DataFrame(columns=['Close', 'Date'],
                  index=shreyas)

predy = lst_output.reshape(period,)
df['Close'] = predy
df['Date'] = shreyas

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'],
                         y=df['Close'],
                         marker={'color': 'blue'},
                         name="stock_close"))
fig.layout.update(
    title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
# fig.update_traces(line_color='purple')
st.plotly_chart(fig)
