# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:25:28 2021

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
    t_host = "15.207.18.6"
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

def get_current_df():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=51550
        df = market_df[market_df['marketIndex']==mid]
        df = df.sort_values('timestamp')     
        df['type'] = df['type'].fillna('Shorter')   
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
            
        intial_liquidity = df['participationValue'][df['playerAddress']==mc_address].sum()/2
        intial_liquidity_2x = 2*intial_liquidity
        
        option_count = df.groupby('optionNumber')['participationValue'].sum().reset_index()
        option_count = option_count.sort_values('participationValue', ascending=False)
        algo_option = option_count['optionNumber'].iloc[0]
        
        user_df = df[~df['bot'].isin(['Bot', 'MC'])] 
        option_1_bet = user_df['participationValue'][user_df['optionNumber']==1].sum()
        option_2_bet = user_df['participationValue'][user_df['optionNumber']==2].sum()
        
        user_amt_diff = abs(option_1_bet-option_2_bet)
        user_amt_diff_3x = 3*user_amt_diff 
        
        participationValue_algo = min(user_amt_diff_3x, intial_liquidity_2x)
        
        df['position'] = df.apply(lambda x : (x['participationValue']/0.5)*mc_multiplier if x['bot']=='MC' else x['position'], axis=1)
        
        df['fees'] = df.apply(lambda x : 0 if x['playerAddress']==mc_address else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
        total_position = df['position'][df['returnInPlot']>0].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['returnInPlot']>0 else 0, axis=1)
        
        bot_mc_return = df['reward'][df['bot'].isin(['Bot','MC'])].sum()
        in_pnl = df['participationValue'][df['bot'].isin(['Bot','MC'])].sum()  
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl
        

        bot_df =  df[df['bot']=='Bot']
        # bot_option = bot_df[bot_df['playerAddress']!=mc_address]
        
        try:
            bot_option = bot_df['optionNumber'].iloc[0]      
        except:
            bot_option=0
    
            
        
        participationValue_bot = bot_df['participationValue'][bot_df['bot']=='Bot'].sum()
        return_bot = bot_df['reward'][bot_df['bot']=='Bot'].sum()
        
        # market_id = bot_df['marketIndex'].iloc[0]
        result={
                "optionToBeChosen_c":algo_option,
                "optionChosenByBot_c":bot_option,
                "participationValueAlgo_c":participationValue_algo,
                "participationValueBot_c":participationValue_bot,
                "returnInPlotBot_c":return_bot,
                "pnl_c":pnl
                }
        
        total_df = total_df.append(df, ignore_index=True)
        # total_d[mid] = result
        total_d[mid] = pnl
        
        # current_df = pd.DataFrame(total_d).T.reset_index()
    # return current_df
    return total_d, total_df


def get_algo_df():
    total_d = dict()
    total_df = pd.DataFrame()

    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        
        df = market_df[market_df['marketIndex']==mid]
        df = df.sort_values('timestamp')     
        df['type'] = df['type'].fillna('Shorter')
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
            
        intial_liquidity = df['participationValue'][df['playerAddress']==mc_address].sum()/2
        intial_liquidity_2x = 2*intial_liquidity
        
        option_count = df.groupby('optionNumber')['participationValue'].sum().reset_index()
        option_count = option_count.sort_values('participationValue', ascending=False)
        algo_option = option_count['optionNumber'].iloc[0]
        
        user_df = df[~df['playerAddress'].isin(bot_address)]
        
        option_1_bet = user_df['participationValue'][user_df['optionNumber']==1].sum()
        option_2_bet = user_df['participationValue'][user_df['optionNumber']==2].sum()
        user_amt_diff = abs(option_1_bet-option_2_bet)
        user_amt_diff_3x = 3*user_amt_diff 
        
        participationValue_algo = min(user_amt_diff_3x, intial_liquidity_2x)
        winning_option = df.groupby('optionNumber')['returnInPlot'].sum().reset_index()
        winning_option = winning_option['optionNumber'][winning_option['returnInPlot']>0].iloc[0]
        
        df['participationValue'] = df.apply(lambda x : participationValue_algo if x['bot']=='Bot' else x['participationValue'],axis=1)
        df['position'] = df.apply(lambda x : x['participationValue']*0.98/0.5 if x['bot']=='Bot' else x['position'],axis=1)
        
        if winning_option!=algo_option:
            df['optionNumber'] = df.apply(lambda x : algo_option if x['bot']=='Bot' else x['optionNumber'],axis=1)
            df['returnInPlot'] = df.apply(lambda x : x['returnInPlot'] if (x['optionNumber']==winning_option) else 0,axis=1)
        
        df['position'] = df.apply(lambda x : (x['participationValue']/0.5)*mc_multiplier if x['bot']=='MC' else x['position'], axis=1)
 
        df['fees'] = df.apply(lambda x : 0 if x['playerAddress']==mc_address else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
        total_position = df['position'][df['returnInPlot']>0].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['returnInPlot']>0 else 0, axis=1)

        bot_mc_return = df['reward'][df['bot'].isin(['Bot','MC'])].sum()
        in_pnl = df['participationValue'][df['bot'].isin(['Bot','MC'])].sum()  
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl

        bot_df =  df[df['playerAddress'].isin(bot_address)]
        bot_option = bot_df[bot_df['playerAddress']!=mc_address]
        
        try:
            bot_option = bot_option['optionNumber'].iloc[0]      
        except:
            bot_option=0
    
            
        
        participationValue_bot = bot_df['participationValue'][bot_df['playerAddress']!=mc_address].sum()
        return_bot = bot_df['reward'][bot_df['playerAddress']!=mc_address].sum()
        
        # market_id = bot_df['marketIndex'].iloc[0]
        result={
                "optionToBeChosen_a":algo_option,
                "optionChosenByBot_a":bot_option,
                "participationValueAlgo_a":participationValue_algo,
                "participationValueBot_a":participationValue_bot,
                "returnInPlotBot_a":return_bot,
                "pnl_a":pnl
                }
        
        total_df = total_df.append(df, ignore_index=True)
        # total_d[mid] = result
        total_d[mid] = pnl
        
        # algo_df = pd.DataFrame(total_d).T.reset_index()
    # return algo_df    
    return total_d, total_df

    
def get_neutral_df():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=51206
        df = market_df[market_df['marketIndex']==mid]
        df = df[df['bot']!='Bot']
        df = df.sort_values('timestamp')     
        df['type'] = df['type'].fillna('Shorter')   
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
        
        df['position'] = df.apply(lambda x : (x['participationValue']/0.5)*mc_multiplier if x['bot']=='MC' else x['position'], axis=1)

        df['fees'] = df.apply(lambda x : 0 if x['playerAddress']==mc_address else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['participationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
        total_position = df['position'][df['returnInPlot']>0].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['returnInPlot']>0 else 0, axis=1)

        bot_mc_return = df['reward'][df['bot'].isin(['Bot','MC'])].sum()
        in_pnl = df['participationValue'][df['bot'].isin(['Bot','MC'])].sum()  
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl

        bot_df =  df[df['playerAddress'].isin(bot_address)]
        bot_option = bot_df[bot_df['playerAddress']!=mc_address]
        
        try:
            bot_option = bot_option['optionNumber'].iloc[0]      
        except:
            bot_option=0
    
        participationValue_bot = bot_df['participationValue'][bot_df['playerAddress']!=mc_address].sum()
        return_bot = bot_df['reward'][bot_df['playerAddress']!=mc_address].sum()
        
        # market_id = bot_df['marketIndex'].iloc[0]
        # result={
        #         "optionToBeChosen_n":algo_option,
        #         "optionChosenByBot_n":bot_option,
        #         "participationValueAlgo_n":participationValue_algo,
        #         "participationValueBot_n":participationValue_bot,
        #         "returnInPlotBot_n":return_bot,
        #         "pnl_n":pnl
        #         }
        
        total_df = total_df.append(df, ignore_index=True)
        # total_d[mid] = result
        total_d[mid] = pnl
        
        # neutral_df = pd.DataFrame(total_d).T.reset_index()
    # return neutral_df 
    return total_d, total_df
          

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
    
    klines = client.get_historical_klines(symbol=sym, interval=interval,start_str=start, end_str=end)

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
    strike = strike_price['strike'][strike_price['marketIndex']==mid].iloc[0]
    return strike

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
        # mid=50401
        strike = get_strike(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        #TODO: Remove Bot
        #df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df = df.sort_values('timestamp')
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
        #df['vop']  = df['vop']+0.02
        
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

def vop_tsm_1():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=50401
        strike = get_strike(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        #TODO: Remove Bot
        #df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df = df.sort_values('timestamp')
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
        # df['vop'] = df.apply(lambda x : x['vop'] if x['optionNumber']==1 else 1-x['vop'], axis=1)
        # df['vop'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop'], axis=1)
        
        df = pd.merge_asof(df.sort_values('move_sd'), phi, on='move_sd', direction='nearest')
        df = df.sort_values('timestamp')
        
        # df['vop'] = 1 - df['vop']
        df['vop'] = df.apply(lambda x : 1-x['vop'] if x['itm']==1 else x['vop'],axis=1)
        df['vop'] = df.apply(lambda x : 1-x['vop'] if (x['itm']!=x['optionNumber']) else x['vop'],axis=1)
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

def voppr4():
    total_d = dict()
    total_df = pd.DataFrame()
    
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=53598
        strike = get_strike(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        #TODO: Remove Bot
        #df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df = df.sort_values('timestamp')
        df['datetime'] = df['transactionTime'].map(lambda x: (x-datetime.timedelta(minutes=1)).replace(second=0, tzinfo=None))
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
        df['fees'] = df.apply(lambda x : 0 if x['bot']=='MC' else x['participationValue']*0.02, axis=1)
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        
        df['option1'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==1 else 0, axis=1) 
        df['option2'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==2 else 0, axis=1) 
        
        df['cumm_1'] = df['option1'].cumsum().shift(1).fillna(0)
        df['cumm_2'] = df['option2'].cumsum().shift(1).fillna(0)
        
        df['new_pos'] = df.apply(lambda x : vop4_pos(x['actualParticipationValue'], x['cumm_1'], x['cumm_2']) if x['option1']!=0 else vop4_pos(x['actualParticipationValue'], x['cumm_2'], x['cumm_1']), axis=1)
        df['vop'] = df['actualParticipationValue']/df['new_pos']
        
        #TODO: Add 2 cents
        # df['vop']  = df['vop']+0.02
        
        df['vop'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop'], axis=1)
        
        df['mc_share_of_fees'] = df['fees']*0.4

        # df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
        # df['new_pos'] = df['actualParticipationValue']/df['vop']
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

def hybrid():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid=53598
        strike = get_strike(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        #TODO: Remove Bot
        #df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df = df.sort_values('timestamp')
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
       
        df['vop_tsm'] = df.apply(lambda y : scipy.stats.norm.cdf(x=y['strike_price'], loc=y['close'], scale=y['v_est']), axis=1)
        # df['vop'] = df.apply(lambda x : x['vop'] if x['itm']==1 else 1-x['vop'], axis=1)
        df['vop_tsm'] = df.apply(lambda x : x['vop_tsm'] if x['optionNumber']==1 else 1-x['vop_tsm'], axis=1)
        
        df['fees'] = df.apply(lambda x : 0 if x['playerAddress']==mc_address else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['participationValue'] if x['returnInPlot']==0 else 0, axis=1)
         
        df['option1'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==1 else 0, axis=1) 
        df['option2'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==2 else 0, axis=1) 
        
        df['cumm_1'] = df['option1'].cumsum().shift(1).fillna(0)
        df['cumm_2'] = df['option2'].cumsum().shift(1).fillna(0)
        
        df['pos_pr4'] = df.apply(lambda x : vop4_pos(x['actualParticipationValue'], x['cumm_1'], x['cumm_2']) if x['option1']!=0 else vop4_pos(x['actualParticipationValue'], x['cumm_2'], x['cumm_1']), axis=1)
        df['vop_pr4'] = df['actualParticipationValue']/df['pos_pr4']
        df['vop_pr4'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop_pr4'], axis=1)
        
        df['vop'] = df[['vop_tsm', 'vop_pr4']].values.max(1)
        df['vop'] = df.apply(lambda x: min(x['vop_tsm'], x['vop_pr4']) if x['vop_tsm']<0.5 else x['vop'], axis=1)
        
        #TODO: Add 2 ccents
        # df['vop'] = df['vop'] + 0.02
        
        df['vop'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop'], axis=1)

        # df['fees'] = df.apply(lambda x : 0 if x['bot']=='MC' else x['participationValue']*0.02, axis=1)
        # df['mc_share_of_fees'] = df['fees']*0.4

        # df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['returnInPlot']==0 else 0, axis=1)
        
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
    
    mc_address = '0x6b8f9c3f66842a82b80d2a24daf53d6df311d59c'
    mc_multiplier = 1.1
        
    strike_price = pd.read_csv(r"C:\Somish\plotx\strike_price_from_9Nov.csv")
    strike_price['strike'] = strike_price['neutralBaseValue']/100000000
    strike_price = strike_price[strike_price['startTime']>1637798400]
    market_ids = tuple(strike_price['marketIndex'].tolist())
    
    phi = pd.read_excel(r"C:\Somish\plotx\Anshul_PlotX3 LP3V1.xlsx", sheet_name='Phi Values', names=['vop','move_sd'])
    
    stats_start = datetime.datetime(year=2021, month=11, day=4)
    stats = stats_24_hours(stats_start)
    
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
    
    d, df = get_current_df()
    df.to_csv(r"{}\current_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\current_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = get_algo_df()
    df.to_csv(r"{}\algo_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\algo_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = get_neutral_df()
    df.to_csv(r"{}\neutral_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\neutral_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = vop_tsm()
    df.to_csv(r"{}\voptsm_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\voptsm_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = vop_tsm_1()
    df.to_csv(r"{}\voptsm_raw_1_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\voptsm_pnl_1_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = voppr4()
    df.to_csv(r"{}\voppr4_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\voppr4_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)
    
    d, df = hybrid()
    df.to_csv(r"{}\vophybrid_raw_{}.csv".format(output_dir,mc_multiplier), index=False)
    df_pnl = pd.DataFrame([d]).T.reset_index()
    df_pnl.columns=['marketIndex','pnl']
    df_pnl.to_csv(r"{}\vophybrid_pnl_{}.csv".format(output_dir,mc_multiplier), index=False)