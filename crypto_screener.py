#!/home/coil/anaconda3/bin/python3


import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime

import smtplib
from email.mime.text import MIMEText

# pretty printing of pandas dataframe
#pd.set_option('expand_frame_repr', False)

# ############################################ #
#           __crypto+specific_part__           #
# ############################################ #

# GET CURRENT PRICE DATA
def get_current_data(from_sym='BTC', to_sym='USD', exchange=''):
    url = 'https://min-api.cryptocompare.com/data/price'

    parameters = {'fsym': from_sym,
                  'tsyms': to_sym }

    if exchange:
        print('exchange: ', exchange)
        parameters['e'] = exchange

    # response comes as json
    response = requests.get(url, params=parameters)
    data = response.json()
    print(data)

    return data

# this is just testing function to see if we get the data
def get_hist_data(from_sym='BTC', to_sym='USD', timeframe = 'day', limit=2000, aggregation=1, exchange=''):

    baseurl = 'https://min-api.cryptocompare.com/data/v2/histo'
    baseurl += timeframe

    parameters = {'fsym': from_sym,
                  'tsym': to_sym,
                  'limit': limit,
                  'aggregate': aggregation}
    if exchange:
        print('exchange: ', exchange)
        parameters['e'] = exchange

    print('baseurl: ', baseurl)
    print('timeframe: ', timeframe)
    print('parameters: ', parameters)

    # response comes as json
    response = requests.get(baseurl, params=parameters)

    data = response.json()['Data']['Data']
    #print(data)

    return data

def data_to_dataframe(data):
    #data from json is in array of dictionaries
    df = pd.DataFrame.from_dict(data)

    # time is stored as an epoch, we need normal dates
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    print(df.tail())

    return df


def plot_data(df, cryptocurrency, target_currency):
    # got his warning because combining matplotlib
    # and time in pandas converted from epoch to normal date
    # To register the converters:
    # 	>>> from pandas.plotting import register_matplotlib_converters
    # 	>>> register_matplotlib_converters()
    #  warnings.warn(msg, FutureWarning)

    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    plt.figure(figsize=(15,5))
    plt.title('{} / {} price data'.format(cryptocurrency, target_currency))
    plt.plot(df.index, df.close)
    plt.legend()
    plt.show()

    return None

# ############################################ #
#           __general_part__                   #
# ############################################ #

username = 'coin.market.cap.000@gmail.com'
password = ''


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


def construct_df(data):
    #get data from yahoo API
    df = data_to_dataframe(data)
    # compute both types of moving averages
    for i in range(50, 250, 50):
        #print(i)
        df['SMA_{}'.format(i)] = computeSMA(df['close'], i)
    for i in range(50, 250, 50):
        #print(i)
        df['EMA_{}'.format(i)] = computeEMA(df['close'], i)

    return df



def send_email(data_rsi, data_200_ema, data_50_ema, data_200_ema_vicinity, username, password):

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



    msg_body = msg_body_rsi + msg_body_200_ema + msg_body_50_ema + msg_body_200_ema_vicinity


    message = MIMEText(msg_body, "plain")
    # treat message as dictionary
    message['subject'] = 'crypto event'
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



# ############################################ #
#           __execution_part__                 #
# ############################################ #


tickers = ['BTC', 'ETH', 'LTC', 'NEO', 'XLM', 'XRP', 'OMG', 'LINK',   \
           'ADA', 'NANO', 'MIOTA', 'DOT', 'EOS', 'ICX', 'ARK']
target_currency = 'USD'

# implement lists as dictionaries for clarity
signal = {}
signal['RSI'] = []
signal['EMA_200'] = []
signal['EMA_50'] = []
signal['EMA_200_vicinity'] = []





for ticker in tickers:

    try:

        # quickly test data aquisition
        #get_current_data(ticker, target_currency, 'coinbase')

        #data = get_hist_data(ticker, target_currency, 'day', 1000, exchange='bitfinex')
        data = get_hist_data(ticker, target_currency, 'day', 1000)



        #plotting to be removed when running on server
        #plot_data(df, ticker, target_currency)



        # df = get_data(ticker)       #just gets data
        df = construct_df(data)     #gets data and adds MAs to the df (implement RSI later)
        #adds RSI column to dataframe
        df['RSI'] = computeRSI(df['close'], 14)
        # RSI <= 30 is long signal
        # if last day RSI data (today) is oversold, send mail
        print('ticker:', ticker)
        print('rsi today', df['RSI'].iloc[-1])

        ## RSI day before <= threshold and RSI today above - long signal
        ##if (df['RSI'].iloc[-2] < 30 and df['RSI'].iloc[-1] >= 30):
        ##    long_list.append(ticker)

        #s __signal_conditions__
        if (df['RSI'].iloc[-1] <= 30):
            signal['RSI'].append(ticker)

        # was below 200 EMA few days ago but today is above 200 EMA
        # possible long
        if (
            (df['EMA_200'].iloc[-5] > df['close'].iloc[-5]) and
            (df['EMA_200'].iloc[-1] < df['close'].iloc[-1])
           ):
            signal['EMA_200'].append(ticker)

        # price in vicinity 50 EMA
        # possible long or at least alert
        if (
            ((df['EMA_50'].iloc[-1] / df['close'].iloc[-1]) >= 0.97) and
            ((df['EMA_50'].iloc[-1] / df['close'].iloc[-1]) <= 1.03)
           ):
            signal['EMA_50'].append(ticker)

        # price in vicinity 200 EMA
        # possible long or at least alert
        if (
            ((df['EMA_200'].iloc[-1] / df['close'].iloc[-1]) >= 0.97) and
            ((df['EMA_200'].iloc[-1] / df['close'].iloc[-1]) <= 1.03)
           ):
            signal['EMA_200_vicinity'].append(ticker)



    except Exception as e:
        print("type error: " + str(e))


if ( len(signal['RSI']) > 0 )     or        \
   ( len(signal['EMA_200']) > 0 ) or        \
   ( len(signal['EMA_50']) > 0 )  or        \
   ( len(signal['EMA_200_vicinity']) > 0 )  :
    rsi_str     = ' '.join(map(str, signal['RSI']))
    ema_200_str = ' '.join(map(str, signal['EMA_200']))
    ema_50_str  = ' '.join(map(str, signal['EMA_50']))
    ema_200_vicinity_str = ' '.join(map(str, signal['EMA_200_vicinity']))

    send_email(rsi_str, ema_200_str, ema_50_str, ema_200_vicinity_str, username, password)

