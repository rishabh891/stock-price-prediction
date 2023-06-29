import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import re
import numpy as np
stock_data=pd.read_csv("stockticker_new2.csv")
company_list=stock_data['company'].tolist()
#st.dataframe(stock_data)
st.sidebar.title("Stock Price Predictor")
selected_company=st.sidebar.selectbox("Choose a stock",company_list)
selected_ticker=stock_data[stock_data['company']==selected_company]['Ticker'].values
selected_ticker = ''.join(selected_ticker)

min_date = datetime.today() - timedelta(days=7500)

start_date = st.sidebar.date_input("Starting Date",min_value=min_date)
end_date = st.sidebar.date_input("End Date")

df = yf.download(selected_ticker, start=start_date, end=end_date)
df=df.reset_index()

analysis_button=st.sidebar.button("Get Analysis")
if analysis_button:
    st.header(''.join(stock_data[stock_data['company'] == selected_company]['Name'].values))
    st.dataframe(df,width=2000,height=500)
    st.subheader("Closing Price vs Time Chart")
    base1 = alt.Chart(df).encode(
        x='Date'
    )
    line11 = base1.mark_line(color='blue').encode(
        y='Close'
    )
    chart1 = alt.layer(line11).resolve_scale(y='independent')

    # Render the chart using Streamlit
    st.altair_chart(chart1, use_container_width=True)




    st.subheader("Closing Price vs Time Chart with 100days moving average")
    ma100=df.Close.rolling(100).mean()
    df['100_days_moving_average']=ma100
    base2 = alt.Chart(df).mark_bar().encode(
        x='Date'
    )
    line12 = base2.mark_line(color='blue').encode(
        y='Close'
    )

    line22 = base2.mark_line(color='red').encode(
        y='100_days_moving_average'
    )

    # Combine the line plots
    chart2 = alt.layer(line12, line22).resolve_scale(y='independent')

    # Render the chart using Streamlit
    st.altair_chart(chart2, use_container_width=True)

    st.subheader("Closing Price vs Time Chart with 100days MA & 200days MA")
    ma200 = df.Close.rolling(200).mean()
    df['200_days_moving_average'] = ma200
    line23=base2.mark_line(color='green').encode(
        y='200_days_moving_average'
    )
    chart3 = alt.layer(line12, line22,line23).resolve_scale(y='independent')

    # Render the chart using Streamlit
    st.altair_chart(chart3, use_container_width=True)

    df1 = df.reset_index()['Close']
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    data_training = df1[0:int(len(df1) * 0.70), :]
    data_testing = df1[int(len(df1) * 0.70):int(len(df1)), :1]

    def create_dataset(dataset,time_step):
        dataX,dataY=[],[]
        for i in range(len(dataset)-time_step-1):
            dataX.append(dataset[i:i+time_step,0])
            dataY.append(dataset[i+time_step,0])
        return np.array(dataX),np.array(dataY)

    X_train,y_train=create_dataset(data_training,100)
    X_test,y_test=create_dataset(data_testing,100)

    X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
    X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    from keras.layers import Dense, Dropout, LSTM
    from keras.models import Sequential

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1))

    model.compile(optimizer='Adam', loss='mse')
    model.fit(X_train, y_train, epochs=10)

    x_input = df1[len(df1) - 100:].reshape(1, -1)
    temp_input = list(x_input)

    temp_input = temp_input[0].tolist()

    lst_output = []
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape(1, 100, 1)
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape(1, 100, 1)
            yhat = model.predict(x_input)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1
    y_fut = scaler.inverse_transform(lst_output)
    y_fut = y_fut.reshape(-1)

    st.header("Next 30 days prediction")

    date=[]
    for i in range(30):
        date.append(end_date+timedelta(days=i+1))
    dfn = pd.DataFrame({'Date':date})
    dfn['Close']=y_fut
    st.dataframe(dfn,width=2000,height=500)
    dates_behind=[]
    for i in range(100):
        dates_behind.append(end_date-timedelta(days=100-i))

    dfx = pd.DataFrame({
        'Date': df['Date'].dt.date,
        'Close': df['Close']
    })
    concatenated_df = pd.concat([dfx, dfn], axis=0)
    concatenated_df=concatenated_df.reset_index()
    st.header("Predicted Closing Price vs Time")

    chart = alt.Chart(concatenated_df).mark_line().encode(
        x='Date',
        y='Close'
    )

    # Render the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)












