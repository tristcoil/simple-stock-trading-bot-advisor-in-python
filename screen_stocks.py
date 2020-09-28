#!/home/coil/anaconda3/bin/python3

# --------------------------------------------------------------
# this script only generates signals to go long
# it is intended to generate buy signals for long term investing
# --------------------------------------------------------------

# my program to run on google free tier VM
# stock screener

#Installing collected packages: multitasking, numpy, python-dateutil, pandas, yfinance
#  WARNING: The scripts f2py, f2py3 and f2py3.8 are installed in '/home/coil/.local/bin' which is not on PATH.
#  Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#  WARNING: The script sample is installed in '/home/coil/.local/bin' which is not on PATH.

#optional installations:
#!pip install yfinance --upgrade --no-cache-dir
# or maybe
#!pip3 install yfinance --upgrade --no-cache-dir
#!pip3 install pandas_datareader



# create lock file quickly to prevent cron spinning infinite number of
# script instances
import sys
import os

lockfile = '/home/coil/scripts/script.lock'
if not os.path.exists(lockfile):
    with open(lockfile, 'w'): pass
else:
    print('lockfile exists, exiting')
    sys.exit(1)



# ___library_import_statements___
import pandas as pd

# for pandas_datareader, otherwise it might have issues, sometimes there is some version mismatch
pd.core.common.is_list_like = pd.api.types.is_list_like

# make pandas to print dataframes nicely
pd.set_option('expand_frame_repr', False)

import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

#newest yahoo API
import yfinance as yahoo_finance

import smtplib
from email.mime.text import MIMEText



#                           ___variables___
# ------------------------------------------------------------------------------

#tickers = ['AAPL','AMZN', 'GOOG', 'FB', 'MSFT', 'BPY', 'XOM']
#ticker = 'AAPL'

with open('/home/coil/scripts/stock_list.txt', 'r') as f:
    # list in following format
    # MMM 3M Company  Industrials Industrial Conglomerates
    #tickers = [f.read().splitlines()[0] for line in f]
    tickers = [line.split()[0] for line in f]


start_time = datetime.datetime(2017, 10, 1)
#end_time = datetime.datetime(2019, 1, 20)
end_time = datetime.datetime.now().date().isoformat()     # today

username = 'coin.market.cap.000@gmail.com'
password = ''




#                       __function_definitions__
# ------------------------------------------------------------------------------

def get_data(ticker):
    # yahoo gives only daily historical data
    attempts = 0
    connected = False
    while not connected:
        try:
            ticker_df = web.get_data_yahoo(ticker, start=start_time, end=end_time)
            connected = True
            print('connected to yahoo')
        except Exception as e:
            print("type error: " + str(e))
            time.sleep( 5 )
            attempts += 1
            if attempts >= 10:
                connected = True
            pass

    # use numerical integer index instead of date
    #ticker_df = ticker_df.reset_index()
    print(ticker_df.head(5))

    return ticker_df

# compute RSI values
def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff

    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]

    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]

    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi


def computeSMA(data, window):
    # simple moving average
    sma = data.rolling(window=window).mean()
    return sma

def computeEMA(data, span):
    # simple moving average
    ema = data.ewm(span=span, adjust=False).mean()
    return ema


def construct_df(ticker):
    #get data from yahoo API
    df = get_data(ticker)
    # compute both types of moving averages
    for i in range(50, 250, 50):
        #print(i)
        df['SMA_{}'.format(i)] = computeSMA(df['Adj Close'], i)
    for i in range(50, 250, 50):
        #print(i)
        df['EMA_{}'.format(i)] = computeEMA(df['Adj Close'], i)

    return df


def stochastic(data, k_window, d_window, window):
    # input to function is one column from df
    # containing closing price or whatever value we want to extract K and D from

    min_val  = data.rolling(window=window, center=False).min()
    max_val = data.rolling(window=window, center=False).max()

    stoch = ( (data - min_val) / (max_val - min_val) ) * 100

    K = stoch.rolling(window=k_window, center=False).mean()
    #K = stoch

    D = K.rolling(window=d_window, center=False).mean()

    return K, D



def resample(df):
    # weekly timeframe aggregation
    agg_dict = {'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Adj Close': 'last',
                'Volume': 'mean'}

    # resampled dataframe
    # 'W' means weekly aggregation
    df_res = df.resample('W').agg(agg_dict)

    return df_res




def send_email(data_rsi, data_200_ema, data_50_ema, data_200_ema_vicinity, data_weekly_stochRSI, username, password):

    smtp_ssl_host = 'smtp.gmail.com'
    smtp_ssl_port = 465
    sender = 'coin.market.cap.000@gmail.com'
    receiver = 'michal.vasulka@yahoo.com'

    # implicitly joined string
    msg_body_rsi = ("stock ticker RSI around 30 \n"
                "possible long entry \n"
                "ticker/s: \n"
                 + data_rsi + "\n\n")

    msg_body_200_ema = ("went above 200 EMA recently \n"
                "possible long entry \n"
                "ticker/s: \n"
                 + data_200_ema + "\n\n")

    msg_body_50_ema = ("in vicinity of 50 EMA \n"
                "alerting \n"
                "ticker/s: \n"
                 + data_50_ema + "\n\n")

    msg_body_200_ema_vicinity = ("in vicinity of 200 EMA \n"
                "strong alert - support/resistance \n"
                "ticker/s: \n"
                 + data_200_ema_vicinity + "\n\n")

    msg__body_weekly_stochRSI = ("weekly_stochRSI long \n"
                "rather strong alert \n"
                "ticker/s: \n"
                 + data_weekly_stochRSI + "\n\n")


    msg_body = msg_body_rsi + msg_body_200_ema + msg_body_50_ema    \
               + msg_body_200_ema_vicinity + msg__body_weekly_stochRSI


    message = MIMEText(msg_body, "plain")
    # treat message as dictionary
    message['subject'] = 'stock event'
    message['from']    = sender
    message['to']      = receiver


    # contact gmail server and send mail via my gmail dummy account
    try:
        server = smtplib.SMTP_SSL(smtp_ssl_host, smtp_ssl_port)
        server.login(username, password)
        server.sendmail(sender, receiver, message.as_string())
        server.quit()
        print("Successfully sent email")
    except:
        print("Error: unable to send email")



##                          __main_code_part__
# ------------------------------------------------------------------------------

#rsi_signal = []
#ema_signal = []

# implement lists as dictionaries for clarity
signal = {}
signal['RSI'] = []
signal['EMA_200'] = []
signal['EMA_50'] = []
signal['EMA_200_vicinity'] = []
signal['weekly_stochRSI'] = []

for ticker in tickers:
    try:
        # df = get_data(ticker)       #just gets data
        df = construct_df(ticker)     #gets data and adds MAs to the df (implement RSI later)
        #adds RSI column to dataframe
        df['RSI'] = computeRSI(df['Adj Close'], 14)
        # RSI <= 30 is long signal
        # if last day RSI data (today) is oversold, send mail
        print('ticker:', ticker)
        print('rsi today', df['RSI'].iloc[-1])

        df_res = resample(df)
        df_res['RSI'] = computeRSI(df_res['Adj Close'], 14)
        df_res['K'], df_res['D'] = stochastic(df_res['RSI'], 3, 3, 14)

        ## RSI day before <= threshold and RSI today above - long signal
        ##if (df['RSI'].iloc[-2] < 30 and df['RSI'].iloc[-1] >= 30):
        ##    long_list.append(ticker)

        #s __signal_conditions__
        if (df['RSI'].iloc[-1] <= 30):
            signal['RSI'].append(ticker)

        # was below 200 EMA few days ago but today is above 200 EMA
        # possible long
        if (
            (df['EMA_200'].iloc[-5] > df['Adj Close'].iloc[-5]) and
            (df['EMA_200'].iloc[-1] < df['Adj Close'].iloc[-1])
           ):
            signal['EMA_200'].append(ticker)

        # price in vicinity 50 EMA
        # possible long or at least alert
        if (
            ((df['EMA_50'].iloc[-1] / df['Adj Close'].iloc[-1]) >= 0.98) and
            ((df['EMA_50'].iloc[-1] / df['Adj Close'].iloc[-1]) <= 1.02)
           ):
            signal['EMA_50'].append(ticker)

        # price in vicinity 200 EMA
        # possible long or at least alert
        if (
            ((df['EMA_200'].iloc[-1] / df['Adj Close'].iloc[-1]) >= 0.98) and
            ((df['EMA_200'].iloc[-1] / df['Adj Close'].iloc[-1]) <= 1.02)
           ):
            signal['EMA_200_vicinity'].append(ticker)

        # weekly stochastic RSI oversold signal
        thres = 20 # oversold condition for stochRSI
        # setting benevolent thresholds
        if (
            df_res['K'].iloc[-1] <= thres and
            df_res['D'].iloc[-1] <= thres and
            ((df_res['K'].iloc[-1] / df_res['D'].iloc[-1]) >= 0.80) and
            ((df_res['K'].iloc[-1] / df_res['D'].iloc[-1]) <= 1.20)
           ):
            #print('found something', df_res['K'].iloc[-i], df_res['D'].iloc[-i] )
            signal['weekly_stochRSI'].append(ticker)
        elif ( (df_res['K'].iloc[-1] == 0 ) or ( df_res['D'].iloc[-1] == 0 ) ):
            #print('indicators are zeros', df_res['K'].iloc[-i])
            signal['weekly_stochRSI'].append(ticker)



    except Exception as e:
        print("type error: " + str(e))



if ( len(signal['RSI']) > 0 )              or        \
   ( len(signal['EMA_200']) > 0 )          or        \
   ( len(signal['EMA_50']) > 0 )           or        \
   ( len(signal['EMA_200_vicinity']) > 0 ) or        \
   ( len(signal['weekly_stochRSI']) > 0 ) :
    rsi_str     = ' '.join(map(str, signal['RSI']))
    ema_200_str = ' '.join(map(str, signal['EMA_200']))
    ema_50_str  = ' '.join(map(str, signal['EMA_50']))
    ema_200_vicinity_str = ' '.join(map(str, signal['EMA_200_vicinity']))
    weekly_stochRSI_str = ' '.join(map(str, signal['weekly_stochRSI']))

    send_email(rsi_str, ema_200_str, ema_50_str, ema_200_vicinity_str, weekly_stochRSI_str, username, password)


# lockfile cleanup
os.remove(lockfile)

