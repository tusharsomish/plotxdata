# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:50:15 2021

@author: tusha
"""

import pandas as pd
import datetime
from binance import Client

start_date = datetime.datetime(year=2021, month=11, day=1)
end_date = (datetime.datetime.utcnow()-datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)


client = Client()

n=4000

sym_list = ['BTCUSDT', 'ETHUSDT']
d={}

for sym in sym_list:
    # sym = 'BTCUSDT'
    d[sym] = dict()
    interval = '5m'
    start = start_date.strftime("%d %b, %Y")
    end = end_date.strftime("%d %b, %Y")
    
    klines = client.get_historical_klines(symbol=sym, interval=interval,start_str=start, end_str=end)
    ohlc = pd.DataFrame(klines)
    ohlc = ohlc[[0,1,2,3,4]]
    ohlc[[1,2,3,4]] = ohlc[[1,2,3,4]].apply(pd.to_numeric)
    ohlc.columns=['datetime', 'open', 'high', 'low', 'close']
    ohlc['datetime'] = pd.to_datetime(ohlc['datetime'], unit='ms') 
    ohlc['datetime'] = ohlc['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    
    rapid = []
    i=0
    while i<=len(ohlc):
        try:
            print(i)
            strike_row = ohlc.iloc[i]
            strike_date = strike_row['datetime']
            strike_price = strike_row['close']
            
            last_row = ohlc.iloc[i+3]
            last_date = last_row['datetime']
            last_price = last_row['close']
            
            settle = ohlc.iloc[i+12]
            settle_date = settle['datetime']
            settle_price = settle['close']
            
            rapid.append([strike_date, strike_price, last_date, last_price, settle_date, settle_price])
            
            i+=3
        except:
            break
    
    cols=['strike_date', 'strike_price', 'last_date', 'last_price', 'settle_date', 'settle_price']
    df = pd.DataFrame(rapid, columns=cols)
    df['last-strike'] = df['last_price'] - df['strike_price']
    df['option1'] = df.apply(lambda x : 1 if (x['last_price']-x['strike_price'])<0 else 2, axis=1)
    df['settle-strike'] = df['settle_price'] - df['strike_price']
    df['option2'] = df.apply(lambda x : 1 if (x['settle_price']-x['strike_price'])<0 else 2, axis=1)
    df['option_reversal'] = df.apply(lambda x : True if x['option1']!=x['option2'] else False, axis=1)
    d[sym]['rapid'] = df
    
    bullet = []
    i=0
    while i<=len(ohlc):
        try:
            print(i)
            strike_row = ohlc.iloc[i]
            strike_date = strike_row['datetime']
            strike_price = strike_row['close']
            
            last_row = ohlc.iloc[i+1]
            last_date = last_row['datetime']
            last_price = last_row['close']
            
            settle = ohlc.iloc[i+4]
            settle_date = settle['datetime']
            settle_price = settle['close']
            
            bullet.append([strike_date, strike_price, last_date, last_price, settle_date, settle_price])
            
            i+=1
        except:
            break
    
    cols=['strike_date', 'strike_price', 'last_date', 'last_price', 'settle_date', 'settle_price']
    df = pd.DataFrame(bullet, columns=cols)
    df['last-strike'] = df['last_price'] - df['strike_price']
    df['option1'] = df.apply(lambda x : 1 if (x['last_price']-x['strike_price'])<0 else 2, axis=1)
    df['settle-strike'] = df['settle_price'] - df['strike_price']
    df['option2'] = df.apply(lambda x : 1 if (x['settle_price']-x['strike_price'])<0 else 2, axis=1)
    df['option_reversal'] = df.apply(lambda x : True if x['option1']!=x['option2'] else False, axis=1)
    d[sym]['bullet'] = df
    
    
    cheetah = []
    i=0
    while i<=len(ohlc):
        try:
            print(i)
            strike_row = ohlc.iloc[i]
            strike_date = strike_row['datetime']
            strike_price = strike_row['close']
            
            last_row = ohlc.iloc[i+1]
            last_date = last_row['datetime']
            last_price = last_row['close']
            
            settle = ohlc.iloc[i+5]
            settle_date = settle['datetime']
            settle_price = settle['close']
            
            cheetah.append([strike_date, strike_price, last_date, last_price, settle_date, settle_price])
            
            i+=1
        except:
            break
    
    cols=['strike_date', 'strike_price', 'last_date', 'last_price', 'settle_date', 'settle_price']
    df = pd.DataFrame(cheetah, columns=cols)
    df['last-strike'] = df['last_price'] - df['strike_price']
    df['option1'] = df.apply(lambda x : 1 if (x['last_price']-x['strike_price'])<0 else 2, axis=1)
    df['settle-strike'] = df['settle_price'] - df['strike_price']
    df['option2'] = df.apply(lambda x : 1 if (x['settle_price']-x['strike_price'])<0 else 2, axis=1)
    df['option_reversal'] = df.apply(lambda x : True if x['option1']!=x['option2'] else False, axis=1)
    d[sym]['cheetah'] = df


result_d = dict()

for k,v in d.items():
    result_d[k] = dict()
    for k1,v1 in d[k].items():
        df = v1
        df = df.iloc[:n]
        reversal = df['option_reversal'].value_counts().reset_index()
        reversal = reversal['option_reversal'][reversal['index']==True].iloc[0]
        
        prob = (reversal/n)*100
        
        result_d[k][k1] = prob
        
        
        
writer = pd.ExcelWriter(r"C:\Somish\plotx\option_reveral.xlsx")
for k,v in d.items():
    for k1,v1 in d[k].items():
        df = v1.iloc[:4000]
        df.to_excel(writer, sheet_name=k+'-'+k1, index=False)
writer.save()
writer.close()
        
        



