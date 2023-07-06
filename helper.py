# tools

import pandas as pd
import streamlit as st
st.set_page_config(layout='wide')

from prophet import Prophet

# compute sma5, ma10, ma20
def compute_ma(df):
    """
    sma5, sma10, sma20 = compute_ma(df)
    """
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['sMA10 ']= df['Close'].rolling(10).mean()
    df['SMA20 ']= df['Close'].rolling(20).mean()
    df=df.round(2)
    return df

def compute_rsi(df, time_window):   
    """
    rsi = compute_rsi(df, time_window)
    """
    diff = df.diff(1).dropna()        # diff in one field(one day)
    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    rsi=rsi.round(2)
    return rsi

# define a list of color for list : item is red if increase, green if decrease
def color_negative_red(lst):
    color = len(lst)*['#ef5350']
    ll=lst[1:]-lst[:-1]
    l = ['#ef5350'if l>=0 else '#14b143' for l in ll ]
    color[1:]=l
    return color
    

def get_macd(price, slow, fast, smooth):
    """
    macd_df=get_macd(df['Close'], 26, 12, 9)
    """
    #pd.set_option('display.float_format', '{:.2f}'.format)
    exp1 = price.ewm(span=fast, adjust=False).mean()
    exp2 = price.ewm(span=slow, adjust=False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns={'Close': 'macd'})
    signal = pd.DataFrame(macd.ewm(span=smooth, adjust=False).mean()).rename(columns={'macd': 'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns={0: 'hist'})
    frames = [macd, signal, hist]
    
    df = pd.concat(frames, join='inner', axis=1)
    df=df.round(2)
    return df
    
import numpy as np

def implement_macd_strategy(prices, data):
    """
    buy_price, sell_price, macd_signal = implement_macd_strategy(googl['Close'], googl_macd)
    
    """
    buy_price = np.empty(len(data))
    sell_price = np.empty(len(data))
    macd_signal = np.zeros(len(data))
    signal = 0

    buy_price[:] = np.nan
    sell_price[:] = np.nan

    buy_condition = (data['macd'] > data['signal'])
    sell_condition = (data['macd'] < data['signal'])

    buy_signal = np.where(np.logical_and(buy_condition, signal != 1))
    sell_signal = np.where(np.logical_and(sell_condition, signal != -1))

    buy_price[buy_signal] = prices[buy_signal]
    sell_price[sell_signal] = prices[sell_signal]

    macd_signal[buy_signal] = 1
    macd_signal[sell_signal] = -1

    return buy_price, sell_price, macd_signal    
    
    
def implement_macd_strategy(prices, data):    
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)):
        if data['macd'][i] > data['signal'][i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['macd'][i] < data['signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
            
    return buy_price, sell_price, macd_signal
    
import matplotlib.pyplot as plt

def trading_vis_matplotlib(googl, googl_macd, buy_price, sell_price):
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
    ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)

    ax1.plot(googl['Close'], color='skyblue', linewidth=2, label='GOOGL')
    ax1.plot(googl.index, buy_price, marker='^', color='green', markersize=10, label='BUY SIGNAL', linewidth=0)
    ax1.plot(googl.index, sell_price, marker='v', color='r', markersize=10, label='SELL SIGNAL', linewidth=0)
    ax1.legend()
    ax1.set_title('MACD SIGNALS')
    ax2.plot(googl_macd.index, googl_macd['macd'], color='grey', linewidth=1.5, label='MACD')
    ax2.plot(googl_macd.index, googl_macd['signal'], color='skyblue', linewidth=1.5, label='SIGNAL')

    for i in range(len(googl_macd)):
        if str(googl_macd['hist'][i])[0] == '-':
            ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color='#ef5350')
        else:
            ax2.bar(googl_macd.index[i], googl_macd['hist'][i], color='#26a69a')

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    plt.legend(loc='lower right')
    #plt.show()
    st.pyplot(fig)
    #return fig

def prophet_predict(data,days=7):
    if data.empty:
        st.write("No data available for the target stock symbol")
    else:
        #st.write("Historical Data for", symbol)
        #st.write(stock_data)

        # Prepare the data for Prophet forecasting
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

        # Train the Prophet model
        model = Prophet()
        model.fit(df)

        # Generate future dates for forecasting
        future_dates = model.make_future_dataframe(periods=days)

        # Perform the forecasting
        
        forecast = model.predict(future_dates).round(2)
        forecast = forecast[forecast['ds'].dt.weekday < 5]
               
        # Display the forecasted data
        #st.write("Forecasted Data for the Next Week")
        #st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
        #f_=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]#.tail(10)
        return forecast

def pycaret_lgb_pred(data,days=5):
    data['SMA5']=data.Close.rolling(window=5).mean().round(2)
    data['SMA10']=data.Close.rolling(window=10).mean().round(2)
    data['SMA20']=data.Close.rolling(window=20).mean().round(2)
    #data['MACD']=googl_macd['macd'].values
    data['target']=data['Close'].shift(-days)
    
    train=data[:-days]
    test=data[-days:]
    setup(data = train, target = 'target',session_id=42,fold=5);
    lgb = create_model('lightgbm',fold = 5)
    tuned_lgb= tune_model(lgb, n_iter=1000, optimize='RMSE',fold=5)
    final_lgb_model = finalize_model(tuned_lgb)
    cols=data.columns[:-1]
    pred=predict_model(final_lgb_model,data=test[cols])
    return   list(pred['prediction_label'].round(2))                     
    
    

from pyecharts import options as opts
from pyecharts.charts import Bar, Line,Grid,Kline
from pyecharts.commons.utils import JsCode
from pyecharts.faker import Faker
from streamlit_echarts import st_pyecharts
from pycaret.regression import *

def trading_vis_pyecharts(googl,tick_name,googl_macd, buy_price, sell_price):
    # Create the Line chart for GOOGL price
    sma5=googl['SMA5'].tolist()
    sma10=googl['sMA10 '].tolist()
    sma20=googl['SMA20 '].tolist()
    
    xdata=googl.index.strftime('%Y-%m-%d').tolist()
    ohlc_data=googl[['Open','Close','Low','High']].values.tolist()
    
    line = (
        Line()
        #stock_data.index.strftime('%Y-%m-%d')
        #.add_xaxis(xaxis_data=googl.index.tolist())
        .add_xaxis(googl.index.strftime('%Y-%m-%d').tolist())
        .add_yaxis(
            # define in main py
            series_name=tick_name,
            y_axis=googl["Close"].tolist(),
            color="skyblue",
            linestyle_opts=opts.LineStyleOpts(width=4),
        )
        .add_yaxis(
            series_name="BUY SIGNAL",
            y_axis=buy_price,
            symbol="triangle",
            symbol_size=10,
            color="red",
            linestyle_opts=opts.LineStyleOpts(width=0),
        )
        .add_yaxis(
            series_name="SELL SIGNAL",
            y_axis=sell_price,
            symbol="triangle-down",
            symbol_size=16,
            color="green",
            linestyle_opts=opts.LineStyleOpts(width=0),
            # rotate not work
            label_opts=opts.LabelOpts(
               is_show=True, 
               position="top",
               formatter="{c}",
               rotate=180,  
            ),  
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="MACD SIGNALS"),
            # automatic select the suitable range
            yaxis_opts=opts.AxisOpts(type_="value", min_='dataMin', max_='dataMax'),        
            legend_opts=opts.LegendOpts(),
        )
    )
    kline= (
        Kline()
        .add_xaxis(xdata)
        # only list type works
        .add_yaxis(
            "KD",
            ohlc_data,
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ef232a",
                color0="#14b143",
                border_color="#ef232a",
                border_color0="#14b143",
            ),
        )
    )
    overlap_kline=line.overlap(kline)
    
    line_ma = (
    Line()
    .add_xaxis(googl.index.strftime('%Y-%m-%d').tolist())
    #.add_yaxis('Close', stock_data['Close'], yaxis_index=0)
    #.add_yaxis('RSI', rsi, yaxis_index=1)
    .add_yaxis('SMA5', sma5,is_smooth=True,linestyle_opts=opts.LineStyleOpts(opacity=0.5),)
    .add_yaxis('SMA10', sma10)
    .add_yaxis('SMA20', sma20)   
    #.add_yaxis('Buy', stock_data['Close'][buy_signals], symbol='triangle', symbol_size=10)
    #.add_yaxis('Sell', stock_data['Close'][sell_signals], symbol='triangle-down', symbol_size=10)
    #.add_yaxis('Stop Loss', stop_loss)
    #.add_yaxis('Target Price', target_price, linestyle_opts=opts.LineStyleOpts(type_='dashed'))
    #.add_yaxis('Upper Band', upper)
    #.add_yaxis('Middle Band', middle)
    #.add_yaxis('Lower Band', lower)
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(),#title=f"{stocks[selected_stock]}"),#,pos_top='top', pos_left='center'),
                     xaxis_opts=opts.AxisOpts(type_="category"),
                     yaxis_opts=opts.AxisOpts(
                     grid_index=1,split_number=3,
                     axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                     axistick_opts=opts.AxisTickOpts(is_show=False),
                     splitline_opts=opts.SplitLineOpts(is_show=False),
                     axislabel_opts=opts.LabelOpts(is_show=True),
            ),
                    )
    #.extend_axis(yaxis=opts.AxisOpts(name="RSI", position="right",min_=0,max_=100),)        
    )
    
    # Overlap Kline + Line
    overlap_line = overlap_kline.overlap(line_ma)
    
    # add Volumn bar
    vols= googl["Volume"].values.tolist()
    colors=color_negative_red(googl['Volume'].values)
    #st.write(len(colors),len(vols))
    
    # define bar color
    data_pair=[]
    for k,v,c in zip(googl.index.strftime('%Y-%m-%d').tolist(),vols,colors):
        data_pair.append(
            opts.BarItem(
                name=k,
                value=v,
                itemstyle_opts=opts.ItemStyleOpts(color=c),
        ))
    
    bar_1 = (
        Bar()
        .add_xaxis(googl.index.strftime('%Y-%m-%d').tolist())
        .add_yaxis(
            #"Volumn",vols,
            "Volumn",data_pair,
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                #quantity increase color is red else is green
                #color=colors
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts("Volumns"),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    ) 
    #st.write(bar_1.options['color'])


    # Create the Line chart for MACD and SIGNAL lines
    line_macd = (
        Line()
        #.add_xaxis(xaxis_data=googl_macd.index.tolist())
        .add_xaxis(xaxis_data=googl_macd.index.strftime('%Y-%m-%d').tolist())
        .add_yaxis(
            series_name="MACD",
            y_axis=googl_macd["macd"].tolist(),
            color="grey",
            linestyle_opts=opts.LineStyleOpts(width=1.5),
        )
        .add_yaxis(
            series_name="SIGNAL",
            y_axis=googl_macd["signal"].tolist(),
            color="skyblue",
            linestyle_opts=opts.LineStyleOpts(width=1.5),
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))

        .set_global_opts(
            legend_opts=opts.LegendOpts(pos_right="5%", pos_top="10%"),
        )
    )
    

    # Create the Bar chart for MACD Histogram
    bar = (
        Bar()
        #.add_xaxis(xaxis_data=googl_macd.index.strftime('%Y-%m-%d').tolist())
        .add_xaxis(xaxis_data=list(googl_macd.index.strftime('%Y-%m-%d')))
        .add_yaxis(
            "MACD Histogram",
            googl_macd["hist"].tolist(),
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                    function(params) {
                        if (params.data >= 0) {
                            return '#ef5350';
                        } else {
                            return '#26a69a';
                        }
                    }
                    """
                )
            ),
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            yaxis_opts=opts.AxisOpts(
                grid_index=2,
                split_number=4,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )
    
    grid_chart = Grid()
    
    grid_chart.add(
        overlap_line,
        grid_opts=opts.GridOpts(pos_left="3%", pos_right="1%", height="60%"),
    )
    grid_chart.add(
        bar_1,
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
        ),
    )
    grid_chart.add(
        line_macd.overlap(bar)
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
       ),
        grid_opts=opts.GridOpts(
            pos_left="3%", pos_right="1%", pos_top="80%", height="15%"
        ),
    )


    # Render the charts
    #grid_chart.render()
    #return grid_chart
    
    st_pyecharts(grid_chart,height="800px")

import pandas as pd

def calculate_macd_position(macd_signal):
    position = []
    for i in range(len(macd_signal)):
        if macd_signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
    return position

def calculate_strategy(macd_signal, googl_macd, googl):
    position = calculate_macd_position(macd_signal)

    for i in range(len(googl['Close'])):
        if macd_signal[i] == 1:
            position[i] = 1
        elif macd_signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]

    macd = googl_macd['macd']
    signal = googl_macd['signal']
    close_price = googl['Close']

    macd_signal_df = pd.DataFrame(macd_signal).rename(columns={0: 'macd_signal'}).set_index(googl.index)
    position_df = pd.DataFrame(position).rename(columns={0: 'macd_position'}).set_index(googl.index)

    strategy = pd.concat([close_price, macd, signal, macd_signal_df, position_df], join='inner', axis=1)
    return strategy


from math import floor
from termcolor import colored as cl

def backtest_macd_strategy(investment_value, googl, strategy):
    googl_ret = pd.DataFrame(np.diff(googl['Close'])).rename(columns={0: 'returns'})
    macd_strategy_ret = []

    for i in range(len(googl_ret)):
        try:
            returns = googl_ret['returns'][i] * strategy['macd_position'][i]
            macd_strategy_ret.append(returns)
        except:
            pass

    macd_strategy_ret_df = pd.DataFrame(macd_strategy_ret).rename(columns={0: 'macd_returns'})

    number_of_stocks = floor(investment_value / googl['Close'][0])
    macd_investment_ret = []

    for i in range(len(macd_strategy_ret_df['macd_returns'])):
        returns = number_of_stocks * macd_strategy_ret_df['macd_returns'][i]
        macd_investment_ret.append(returns)

    macd_investment_ret_df = pd.DataFrame(macd_investment_ret).rename(columns={0: 'investment_returns'})
    total_investment_ret = round(sum(macd_investment_ret_df['investment_returns']), 2)
    profit_percentage = floor((total_investment_ret / investment_value) * 100)
    
    result = {
        'total_investment_return': total_investment_ret,
        'profit_percentage': profit_percentage
    }
    
    return result
