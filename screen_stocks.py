#!/home/coil/anaconda3/bin/python3

#my program to run on google free tier VM
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

tickers = ['AAPL','AMZN', 'GOOG', 'FB', 'MSFT', 'BPY', 'XOM']

ticker = 'AAPL'

start_time = datetime.datetime(2017, 10, 1)
#end_time = datetime.datetime(2019, 1, 20)
end_time = datetime.datetime.now().date().isoformat()     # today

username = 'coin.market.cap.000@gmail.com'
password = 'putyourpasswordhere'

#                       __function_definitions__

def get_data(ticker):
    # yahoo gives only daily historical data
    connected = False
    while not connected:
        try:
            ticker_df = web.get_data_yahoo(ticker, start=start_time, end=end_time)
            connected = True
            print('connected to yahoo')
        except Exception as e:
            print("type error: " + str(e))
            time.sleep( 5 )
            pass

    # use numerical integer index instead of date
    ticker_df = ticker_df.reset_index()
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

def send_email(data, username, password):

    smtp_ssl_host = 'smtp.gmail.com'
    smtp_ssl_port = 465
    sender = 'coin.market.cap.000@gmail.com'
    receiver = 'michal.vasulka@yahoo.com'

    # implicitly joined string
    msg_body = ("stock ticker RSI <= 30 \n"
                "possible long entry \n"
                "ticker/s: \n"
                 + data)

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

##                     __main_code_part__
long_list = []

for ticker in tickers:
    df = get_data(ticker)
    #adds RSI column to dataframe
    df['RSI'] = computeRSI(df['Adj Close'], 14)
    # RSI <= 30 is long signal
    # if last day RSI data (today) is oversold, send mail
    print('ticker:', ticker)
    print('rsi today', df['RSI'].iloc[-1])

    if df['RSI'].iloc[-1] <= 30:
        long_list.append(ticker)

if len(long_list) > 0:
    long_str = ' '.join(map(str, long_list))
    send_email(long_str, username, password)
