# https://www.cnblogs.com/WeiRonbbin/p/15871028.html

import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from math import floor
from helper import *
from prophet import Prophet

from pycaret.regression import *

#from streamlit_echarts import st_pyecharts

from datetime import datetime, timedelta
# Define the start date
start_date = datetime(2020, 1, 1)

# Get the current date
current_date = datetime.now()
end_date = current_date.strftime('%Y-%m-%d')

# Calculate the difference in days
days_diff = (current_date - start_date).days


# Define the stock symbols and target prices
stocks = {
    '2308.TW': '台達電',
    '2330.TW': '台積電',
    '2382.TW': '廣達',
    '2498.TW': '宏達電',
    '2603.TW': '長榮',
    '2618.TW': '長榮航空',
    '2634.TW': '漢翔',
    '3583.TW': '辛耘',
    '5222.TW': '全訊',
    '5347.TWO': '世界',
    '6415.TW': '矽力',
    '6510.TWO': '精測',
    '6533.TW': '晶心科',
    '6515.TW': '穎崴',
    '6691.TW': '洋基工程',
    '2027.TW': '大成鋼',
    '2634.TW': '漢翔',
    '3017.TW': '奇鋐',
    '3008.TW': '大立光',
    # Add more stocks and their corresponding symbols here
}

 
# Streamlit configuration
st.title("Stock MACD Analysis")

#symbol = st.text_input("Enter a company", "")
selected_stock = st.selectbox("Select a stock", list(stocks.keys()), format_func=lambda x: f"{x} - {stocks[x]}")
time_period = st.selectbox("Select a time period", ['3 months', '6 months', '1 year', 'All'])
company_input = st.text_input("Enter a company", "")
update_button = st.button("Submit")





    
    
    
if update_button:
   if company_input: 
      tick = yf.download(company_input, start=start_date, end=current_date)
      tick_name = company_input 
   else: 
      tick = yf.download(selected_stock, start=start_date, end=current_date)
      tick_name=selected_stock
   # compute ma
   
   tick = compute_ma(tick)
    
   # Calculate MACD
   tick_macd = get_macd(tick['Close'], slow=26, fast=12, smooth=9)
    
    
   # get the data for pred
   data=tick.copy()
   data['MACD']=tick_macd['macd'].values 

   # Implement MACD strategy
   buy_price, sell_price, macd_signal = implement_macd_strategy(tick['Close'], tick_macd)
    
    
   if time_period == '1 month':
       start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
       ndays=30
   elif time_period == '3 months':
       start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
       ndays=90
   elif time_period == '6 months':
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        ndays=180
   elif time_period == '1 year':
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        ndays=365
   else:
        start_date = "2020-01-01"
        ndays=days_diff
   
   tick=tick.tail(ndays)
   tick_macd=tick_macd.tail(ndays)
   buy_price=buy_price[-ndays:]
   sell_price=sell_price[-ndays:]
   macd_signal=macd_signal[-ndays:]
   #difs,deas,macd = difs_[-ndays:], deas_[-ndays:],macd_[-ndays:] 

   # Visualize MACD signals
   #trading_vis_matplotlib(tick, tick_macd, buy_price, sell_price)
    
   trading_vis_pyecharts(tick, tick_name, tick_macd, buy_price, sell_price) 

   # Calculate position and perform backtest
   tick_stradegy=calculate_strategy(macd_signal, tick_macd, tick)

   investment_value = 100000
   result = backtest_macd_strategy(investment_value, tick, tick_stradegy)

   # Print the backtest result
   st.write("Total investment return:", result['total_investment_return'])
   st.write("Profit percentage:", result['profit_percentage'])

   with st.spinner("Proceeding Forecasted Data for the Next Week ..."):
        pred= prophet_predict(data)
        
        # Display the forecasted data
        pred1 = pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
   
        #data=tick.copy()
        #data['MACD']=tick_macd['macd'].values 
        lgb_pred=pycaret_lgb_pred(data)
        pred1['lgb pred']= lgb_pred
    
        #st.write("Forecasted Data for the Next Week")
        st.write(pred1)
        #st.success(result)    