# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 14:53:06 2022

@author: tusha
"""

import pandas as pd
import psycopg2
import numpy as np
import datetime
from binance import Client
import math
import scipy.stats

def db_conn():
    t_host = "13.127.216.173"
    t_port = "5432"
    t_dbname = "plotx"
    t_user = "ubuntu"
    t_pw = "root"
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    return conn

# def get_marketdata():
#     conn = db_conn()
#     c = conn.cursor()
#     c.execute(f""" SELECT * FROM market_predictions WHERE "transactionTime" > '2021-11-27 12:59' 
#               AND db_timestamp=253402257599 AND status='Settled'
#               AND (type='Shorter' OR "playerAddress"='{mc_address}') """)
#     tuples = c.fetchall()
#     colnames = [desc[0] for desc in c.description]
#     conn.close()
#     df = pd.DataFrame(tuples, columns=colnames)
#     return df

def get_marketdata(market_ids):
    conn = db_conn()
    c = conn.cursor()
    c.execute(f""" SELECT * FROM market_predictions 
              WHERE db_timestamp=253402257599 AND status='Settled'
              AND "marketIndex" in {market_ids} """)
    tuples = c.fetchall()
    colnames = [desc[0] for desc in c.description]
    conn.close()
    df = pd.DataFrame(tuples, columns=colnames)
    return df

def get_botaddress():
    conn = db_conn()
    c = conn.cursor()
    c.execute(""" SELECT * FROM bot_address""")
    tuples = c.fetchall()
    conn.close()
    bot_list = [i[0] for i in tuples]
    return bot_list

def get_timefactor(t):
    return math.sqrt(t/60)

def volatility_intime(std_dev, t_factor):
    return std_dev*t_factor
    
def get_stats(df):
    df['move'] = (df['close']-df['open'])/df['open']
    ltp = df['close'].iloc[-1]
    mean = df['move'].mean()
    std_dev = df['move'].std()
    hourly_v = std_dev*ltp
    return mean, std_dev, hourly_v

def stats_1_hours(start_date):
    start = start_date - datetime.timedelta(hours=100)
    start = start.replace(tzinfo=None)
    start = int((start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    end = datetime.datetime.utcnow()
    end = end.replace(tzinfo=None)
    end = int((end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    stats=dict()
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        stats[symbol] = dict()
        
        ohlc = get_ohlc(symbol, start, end, '1h')
        ohlc['datetime'] = ohlc['datetime'].apply(pd.to_datetime)
    
        i=0
        while i<len(ohlc):
            df = ohlc.iloc[i:i+100]
            date_key = df['datetime'].iloc[-1]
            mean, std_dev, hourly_v = get_stats(df)
            stats[symbol].update({date_key:{'mean':mean, 'std_dev':std_dev, 'hourly_v':hourly_v}})
            i+=1
    return stats

def stats_8_hours(start_date):
    start = start_date - datetime.timedelta(hours=100)
    start = start.replace(tzinfo=None)
    start = int((start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    end = datetime.datetime.utcnow()
    end = end.replace(tzinfo=None)
    end = int((end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    stats=dict()
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        stats[symbol] = dict()
        
        ohlc = get_ohlc(symbol, start, end, '1h')
        ohlc['datetime'] = ohlc['datetime'].apply(pd.to_datetime)
    
        i=0
        while i<len(ohlc):
            df = ohlc.iloc[i:i+100]
            date_key = df['datetime'].iloc[-1]
            mean, std_dev, hourly_v = get_stats(df)
            stats[symbol].update({date_key:{'mean':mean, 'std_dev':std_dev, 'hourly_v':hourly_v}})
            i+=8
    return stats

def stats_24_hours(start_date):
    start = start_date - datetime.timedelta(hours=100)
    start = start.replace(tzinfo=None)
    start = int((start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    end = datetime.datetime.utcnow()
    end = end.replace(tzinfo=None)
    end = int((end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    stats=dict()
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        stats[symbol] = dict()
        
        ohlc = get_ohlc(symbol, start, end, '1h')
        ohlc['datetime'] = ohlc['datetime'].apply(pd.to_datetime)
    
        i=0
        while i<len(ohlc):
            df = ohlc.iloc[i:i+100]
            date_key = df['datetime'].iloc[-1]
            mean, std_dev, hourly_v = get_stats(df)
            stats[symbol].update({date_key:{'mean':mean, 'std_dev':std_dev, 'hourly_v':hourly_v}})
            i+=24
    return stats    

def get_ohlc(sym, start, end, interval):
    
    klines = client.get_historical_klines(symbol=sym, interval=interval, start_str=start, end_str=end)

    df = pd.DataFrame(klines)
    df = df[[0,1,2,3,4]]
    df[[1,2,3,4]] = df[[1,2,3,4]].apply(pd.to_numeric)
    df.columns=['datetime', 'open', 'high', 'low', 'close']
    df['datetime'] = pd.to_datetime(df['datetime'], unit='ms') 
    df['datetime'] = df['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.insert(0, 'symbol', sym)
    
    return df

def daily_stats(start_date):
    start = start_date - datetime.timedelta(hours=100)
    close_start = start.replace(tzinfo=None)
    close_start = int((close_start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    close_end = datetime.datetime.utcnow().replace(tzinfo=None)
    close_end = int((close_end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    stats=dict()
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        stats[symbol] = dict()
        ohlc = get_ohlc(symbol, close_start, close_end, '1h')
        ohlc['datetime'] = ohlc['datetime'].apply(pd.to_datetime)
        i=0
        while i+100 < len(ohlc):
            df = ohlc.iloc[i:i+100]
            if len(df)<100:
                df = ohlc.iloc[-100:]
            date_key = df['datetime'].iloc[-1]#.strftime("%Y-%m-%d %H:%M")
            print(date_key)
            mean, std_dev, hourly_v = get_stats(df)
            stats[symbol].update({date_key:{'mean':mean, 'std_dev':std_dev, 'hourly_v':hourly_v}})
            # stats[symbol][date_key] = {'mean':mean, 'std_dev':std_dev, 'hourly_v':hourly_v}
            i+=100
    return stats

def market_stats(start_date, symbol):
    start = start_date - datetime.timedelta(hours=100)
    start = start.replace(tzinfo=None)
    start = int((start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    end = start_date
    end = int((end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    df = get_ohlc(symbol, start, end, '1h')
    df['datetime'] = df['datetime'].apply(pd.to_datetime)
    
    mean, std_dev, hourly_v = get_stats(df)
    
    return mean,std_dev,hourly_v


    
def get_strike(mid):
    strike1 = strike_price['neutralMinValue'][strike_price['marketIndex']==mid].iloc[0]
    strike2 = strike_price['neutralMaxValue'][strike_price['marketIndex']==mid].iloc[0]

    return strike1,strike2


def itm(diff, option):
    if diff<0 and option==1:
        return 1
    elif diff>0 and option==2:
        return 1
    elif diff<0 and option==2:
        return 2
    elif diff>0 and option==1:
        return 2

def vop4_pos(x, e1, e2):
    try:
        log_value = math.log(1 + (x/e1))
        return x + (e2 * log_value)
    except ZeroDivisionError:
        return 0
    
def vop_tsm():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=53023
        strike1, strike2 = get_strike(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        #TODO: Remove Bot
        if remove_bot:
            df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        # df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        df = df.sort_values('timestamp')

        close_start = ((df['timestamp'].iloc[0])-300)*1000
        close_end = ((df['timestamp'].iloc[-1])-300)*1000

        ohlc = get_ohlc(symbol, str(close_start), str(close_end), '1m')
        ohlc['datetime'] = ohlc['datetime'].apply(pd.to_datetime)
        
        
        df['datetime'] = df['transactionTime'].map(lambda x: x.replace(second=0, tzinfo=None))
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
        
        df = pd.merge(df, ohlc[['symbol', 'datetime', 'close']], on=['symbol','datetime'], how='left')
        
        market_creation_date = df['createdAt'].iloc[0].replace(tzinfo=None)
        
        dates=list(stats[symbol].keys())
        for idx, v in enumerate(dates):
            try:
                if dates[idx] < market_creation_date and dates[idx+1]>market_creation_date:
                    mean = stats[symbol][v]['mean']
                    std_dev = stats[symbol][v]['std_dev']
                    hourly_v = stats[symbol][v]['hourly_v']
                    break
            except IndexError:
                mean = stats[symbol][v]['mean']
                std_dev = stats[symbol][v]['std_dev']
                hourly_v = stats[symbol][v]['hourly_v']
        
        # mean, std_dev, hourly_v = market_stats(market_creation_date, symbol)
        
        df['strike_price'] = strike
        df['v'] = hourly_v
        df['std_dev'] = std_dev
        df['mean'] = mean
        
        df['diff'] =  df['close']-strike
        df['t'] = (df['settleTime'] - df['timestamp'])/60
        df['timefactor'] = (df['t']/60).apply(math.sqrt)
        df['v_est'] = hourly_v*df['timefactor']
        df['move_sd'] = df['diff']/df['v_est']
        
        df['itm'] = df['diff'].apply(lambda x : 1 if x<0 else 2)
       
        # df['vop'] = df.apply(lambda y : scipy.stats.norm.cdf(x=y['strike_price'], loc=y['close'], scale=y['v_est']), axis=1)
        # df['vop'] = df.apply(lambda x : x['vop'] if x['itm']==1 else 1-x['vop'], axis=1)
        # df['vop'] = df.apply(lambda x : x['vop'] if x['optionNumber']==1 else 1-x['vop'], axis=1)

        df = pd.merge_asof(df.sort_values('move_sd'), phi, on='move_sd', direction='nearest')
        df = df.sort_values('timestamp')
        
        # df['vop'] = 1 - df['vop']
        df['vop'] = df.apply(lambda x : 1-x['vop'] if x['itm']==1 else x['vop'],axis=1)
        df['vop'] = df.apply(lambda x : 1-x['vop'] if (x['itm']!=x['optionNumber']) else x['vop'],axis=1)
        
        #TODO: Add 2 cents
        if add_2_cent:
            df['vop']  = df['vop']+0.02
            df['vop'] = df['vop'].apply(lambda x : x if x<1 else 0.98)
            
        df['vop'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop'], axis=1)
        
        df['fees'] = df.apply(lambda x : 0 if x['bot']=='MC' else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4

        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
        df['new_pos'] = df['actualParticipationValue']/df['vop']
        df['new_pos'] = df.apply(lambda x : (x['participationValue']/0.5)*mc_multiplier if x['bot']=='MC' else x['new_pos'], axis=1)
       
        total_position = df['new_pos'][df['returnInPlot']>0].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['new_pos']/total_position)*actual_contribution) if x['returnInPlot']>0 else 0, axis=1)
        
        in_pnl = df['participationValue'][df['bot'].isin(['Bot', 'MC'])].sum()
        bot_mc_return = df['reward'][df['bot'].isin(['Bot', 'MC'])].sum()
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl
        
        total_d[mid] = pnl
        total_df = total_df.append(df, ignore_index=True)
    return total_d, total_df
     
if __name__ == '__main__':
    
    client = Client()
    np.seterr(divide='ignore', invalid='ignore')
    pd.options.mode.chained_assignment = None
    
    mc_address = '0xf076ce4e8eee8995c1c572cfafb3a899c309d118'
    mc_multiplier = 1.1
    remove_bot=False
    add_2_cent=True
        
    strike_price = pd.read_csv(r"C:\Somish\plotx\strike_price_3_bucket.csv")
    strike_price['neutralBaseValue'] = strike_price['neutralBaseValue']/100000000
    strike_price['neutralMinValue'] = strike_price['neutralMinValue']/100000000
    strike_price['neutralMaxValue'] = strike_price['neutralMaxValue']/100000000

    strike_price = strike_price[strike_price['startTime']>1637798400]
    market_ids = tuple(strike_price['marketIndex'].tolist())
    
    phi = pd.read_excel(r"C:\Somish\plotx\Anshul_PlotX3 LP3V1.xlsx", sheet_name='Phi Values', names=['vop','move_sd'])
    
    stats_start = datetime.datetime(year=2021, month=11, day=4)
    stats = stats_1_hours(stats_start)
    
    ohlc=pd.DataFrame()
    close_start = stats_start.replace(tzinfo=None)
    close_start = int((close_start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    close_end = datetime.datetime.utcnow().replace(tzinfo=None)
    close_end = int((close_end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        ohlc1 = get_ohlc(symbol, close_start, close_end, '1m')
        ohlc1['datetime'] = ohlc1['datetime'].apply(pd.to_datetime)
        ohlc = ohlc.append(ohlc1, ignore_index=True)
        
    
    bot_address = get_botaddress()

    market_df = get_marketdata(market_ids)
    market_df['bot'] = market_df['playerAddress'].apply(lambda x : 'Bot' if x in bot_address and x!=mc_address else '')
    market_df['bot'] = market_df.apply(lambda x : 'MC' if x['playerAddress']==mc_address else x['bot'], axis=1)
    market_df['bot'] = market_df.apply(lambda x : 'Player' if x['bot']=='' else x['bot'], axis=1)
    market_df = pd.merge(market_df, strike_price[['marketIndex', 'startTime', 'settleTime', 'assetType']], on='marketIndex', how='left')
     
    output_dir = "C:\Somish\plotx\simulation1"
    
    d, df = vop_tsm()
    df.to_csv(r"{}\voptsm_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\voptsm_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    