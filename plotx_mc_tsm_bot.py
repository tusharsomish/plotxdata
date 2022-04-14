# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 23:54:15 2022

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

def get_stats(df):
    df['move'] = (df['close']-df['open'])/df['open']
    ltp = df['close'].iloc[-1]
    mean = df['move'].mean()
    std_dev = df['move'].std()
    hourly_v = std_dev*ltp
    return mean, std_dev, hourly_v

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

def stats_8_hours(start_date):
    start = start_date - datetime.timedelta(hours=100)
    start = start.replace(tzinfo=None)
    start = int((start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    end = datetime.datetime.utcnow()
    end = end.replace(tzinfo=None)
    end = int((end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    
    stats=dict()
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']:
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

def get_strike(mid):
    strike = strike_price['strike'][strike_price['marketIndex']==mid].iloc[0]
    return strike

def get_winning_option(mid):
    strike = strike_price['winningOption'][strike_price['marketIndex']==mid].iloc[0]
    return strike

def vop4_pos(x, e1, e2):
    try:
        log_value = math.log(1 + (x/e1))
        return x + (e2 * log_value)
    except ZeroDivisionError:
        return 0
    
def get_default():
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid = 81446
        
        winning_option = get_winning_option(mid)
        
        df = market_df[market_df['marketIndex']==mid]
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
        
        df['fees'] = df.apply(lambda x : 0 if x['bot']=='MC' else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']!=winning_option else 0, axis=1)
        
        total_position = df['position'][df['optionNumber']==winning_option].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['optionNumber']==winning_option else 0, axis=1)
        
        bot_mc_return = df['reward'][df['bot'].isin(['Bot','MC', 'MC Bot'])].sum()
        in_pnl = df['participationValue'][df['bot'].isin(['Bot','MC', 'MC Bot'])].sum()  
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl
        
        total_df = total_df.append(df, ignore_index=True)
        total_d[mid] = pnl

    return total_d, total_df
    
def incremental_update(total_liquidity, remove_bot):
    # total_liquidity=12000
    # remove_bot=False
    total_d = dict()
    total_df = pd.DataFrame()
    for idx, mid in enumerate(list(dict.fromkeys(market_df['marketIndex']))):
        print(f'{idx} -> {mid}')
        # mid = 74787
        strike = get_strike(mid)
        winning_option = get_winning_option(mid)
        
        df = market_df[market_df['marketIndex']==mid]
           
        if remove_bot:
            df = df[df['bot']!='Bot']
        
        df['assetType_x'] = df['assetType_x'].bfill(axis=0).ffill(axis=0)
        df['type'] = df['type'].fillna('Shorter')
        
        symbol = df['assetType_y'].iloc[0].replace('/','')+'T'
        df['symbol']=symbol
        
        df['participationValue'] = df['participationValue']/100000000
        df['returnInPlot'] = df['returnInPlot']/100000000
        
        market_start_time = strike_price['startTime'][strike_price['marketIndex']==mid].iloc[0]
        market_creation_date = datetime.datetime.utcfromtimestamp(market_start_time)
       
        df = df.sort_values('timestamp')
        df.reset_index(inplace=True)
        df = df.drop(['index'],axis=1)
        
        # market_creation_date = df['transactionTime'].iloc[0].replace(tzinfo=None, second=0)
        # market_start_time = (market_creation_date - datetime.datetime(1970, 1, 1)).total_seconds()
        
        market_settle_date = strike_price['settleTime'][strike_price['marketIndex']==mid].iloc[0]
               
        df.at[0, 'participationValue'] = (total_liquidity*init_percent)/2
        df.at[1, 'participationValue'] = (total_liquidity*init_percent)/2
        
        df.at[0, 'position'] = (((total_liquidity*init_percent)/2) / 0.5)
        df.at[1, 'position'] = (((total_liquidity*init_percent)/2) / 0.5)
        
        new_add = [market_start_time+300, market_start_time+600, market_start_time+840]
        
        for idx,time in enumerate(new_add):
            
            new_time = datetime.datetime.utcfromtimestamp(time)
            cmp = ohlc['close'][(ohlc['symbol']==symbol) & (ohlc['datetime']==new_time)].iloc[0]
            cmp_diff = cmp - strike
            if cmp_diff <= 0:
                option_list = [2,1]
            elif cmp_diff > 0:
                option_list = [1,2]
            
            for o in option_list:
                d = {'playerAddress':'mc_tsm',
                     'optionNumber':o,
                     'timestamp':time,
                     'settleTime':market_settle_date,
                     'transactionTime':new_time,
                     'bot':'MC Bot',
                     'symbol':symbol,
                     'participationValue':(total_liquidity*increments[idx]),
                     'marketIndex':mid
                     }
                new_time = new_time+datetime.timedelta(seconds=1)
                time = time+1
                df = df.append(d, ignore_index=True)
        
        df = df.sort_values('timestamp')
        df['datetime'] = df['transactionTime'].map(lambda x: x.replace(second=0, tzinfo=None))
        
        df = pd.merge(df, ohlc[['symbol', 'datetime', 'close']], on=['symbol','datetime'], how='left')
        
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
        
        df = pd.merge_asof(df.sort_values('move_sd'), phi, on='move_sd', direction='nearest')
        
        df['vop'] = df.apply(lambda x : 1-x['vop'] if x['itm']==1 else x['vop'],axis=1)
        df['vop'] = df.apply(lambda x : 1-x['vop'] if (x['itm']!=x['optionNumber']) else x['vop'],axis=1)
        
        df['participationValue'] = df.apply(lambda x : x['participationValue']*x['vop'] if x['playerAddress']=='mc_tsm' else x['participationValue'], axis=1)
        df['position'] = df.apply(lambda x : (x['participationValue']/0.5)*mc_multiplier if x['playerAddress']=='mc_tsm' else x['position'], axis=1)
        
        ##################### Participation Algo #####################
        if 'Bot' in df['bot']:
            df = df.sort_values('timestamp')
            df = df.reset_index(drop=True)
            
            bot_index = df.index[df['bot']=='Bot'][0]
            mc_totalbet_1 = df['participationValue'][(df['optionNumber']==1) & (df['bot'].isin(['MC', 'MC Bot']))].sum()
            mc_totalbet_2 = df['participationValue'][(df['optionNumber']==2) & (df['bot'].isin(['MC', 'MC Bot']))].sum()
            
            user_df = df.iloc[:bot_index]
            user_totalbet_1 = user_df['participationValue'][(user_df['optionNumber']==1) & (user_df['bot']=='Player')].sum()
            user_totalbet_2 = user_df['participationValue'][(user_df['optionNumber']==2) & (user_df['bot']=='Player')].sum()
            user_bet_diff = abs(user_totalbet_1-user_totalbet_2)
            
            if user_totalbet_1 >= user_totalbet_2:
                bot_option = 1
            else:
                bot_option = 2
            
            amount = min(total_liquidity, 3*user_bet_diff)
            itm_latest = df['itm'][df['bot']=='Bot'].iloc[-1]
            
            if (mc_totalbet_1>=mc_totalbet_2) and (itm_latest==1):
                amount = amount - (mc_totalbet_1-mc_totalbet_2)
            elif (mc_totalbet_1<mc_totalbet_2) and (itm_latest==2):
                amount = amount - (mc_totalbet_2-mc_totalbet_1)
            else:
                pass
            
            df['participationValue'] = df.apply(lambda x : amount if x['bot']=='Bot' else x['participationValue'], axis=1)
            df['optionNumber'] = df.apply(lambda x : bot_option if x['bot']=='Bot' else x['optionNumber'], axis=1)

        ###############################################################
        
        df['fees'] = df.apply(lambda x : 0 if x['bot'] in ['MC', 'MC Bot'] else x['participationValue']*0.02, axis=1)
        df['mc_share_of_fees'] = df['fees']*0.4
        df['actualParticipationValue'] = df['participationValue']-df['fees']
        df['contributionPool'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']!=winning_option else 0, axis=1)
        
        df = df.sort_values('timestamp')
        if pr4:
            df['option1'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==1 else 0, axis=1) 
            df['option2'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==2 else 0, axis=1) 
            
            df['cumm_1'] = df['option1'].cumsum().shift(1).fillna(0)
            df['cumm_2'] = df['option2'].cumsum().shift(1).fillna(0)
            
            df['pos_pr4'] = df.apply(lambda x : vop4_pos(x['actualParticipationValue'], x['cumm_1'], x['cumm_2']) if x['option1']!=0 else vop4_pos(x['actualParticipationValue'], x['cumm_2'], x['cumm_1']), axis=1)
            df['vop_pr4'] = df['actualParticipationValue']/df['pos_pr4']
            
            if add_2_cent:
                df['vop_pr4']  = df['vop_pr4']+0.02
                df['vop_pr4'] = df['vop_pr4'].apply(lambda x : x if x<1 else 0.98)
                df['vop_pr4'] = df.apply(lambda x : 0.5 if x['bot']=='MC' else x['vop_pr4'], axis=1)
      
            df['vop_pr4'] = df.apply(lambda x : 0.5 if x['bot'] in ['MC'] else x['vop_pr4'], axis=1)
            df['new_pos'] = df['actualParticipationValue']/df['vop_pr4']
            df['new_pos'] = df.apply(lambda x : (x['actualParticipationValue']/0.5)*mc_multiplier if x['bot'] in ['MC'] else x['new_pos'], axis=1)
            df['new_pos'] = df.apply(lambda x : (x['actualParticipationValue']/x['vop_pr4'])*mc_multiplier if x['bot'] in ['MC Bot'] else x['new_pos'], axis=1)
       
        if 'new_pos' in df.columns:
            df['position']=df['new_pos']
            
        total_position = df['position'][df['optionNumber']==winning_option].sum()
        total_contribution = df['contributionPool'].sum()
        mc_share_from_contribution_pool = total_contribution*0.05
        actual_contribution = total_contribution - mc_share_from_contribution_pool
        mc_share_of_fees = df['mc_share_of_fees'].sum()
        
        df['reward'] = df.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['optionNumber']==winning_option else 0, axis=1)
        
        bot_mc_return = df['reward'][df['bot'].isin(['Bot','MC', 'MC Bot'])].sum()
        in_pnl = df['actualParticipationValue'][df['bot'].isin(['Bot','MC', 'MC Bot'])].sum()  
        out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
        pnl = out_pnl - in_pnl
        
        total_df = total_df.append(df, ignore_index=True)
        total_d[mid] = pnl

    return total_d, total_df

if __name__ == '__main__':
    
    client = Client()
    np.seterr(divide='ignore', invalid='ignore')
    pd.options.mode.chained_assignment = None
    
    mc_address = '0x6b8f9c3f66842a82b80d2a24daf53d6df311d59c'
    mc_multiplier = 1.1
    remove_bot_list=[False]
    add_2_cent=True
    pr4=True
    total_liquidity_list = [20000]
    init_percent = 0.2
    increments = [0.25, 0.25, 0.3]

    strike_price = pd.read_csv(r"C:\Somish\plotx\mc_tsm\strike_price_2_bucket.csv")
    strike_price['strike'] = strike_price['neutralBaseValue']/100000000
    # strike_price = strike_price[strike_price['startTime']>1637798400]
    market_ids = tuple(strike_price['marketIndex'].tolist())
    
    phi = pd.read_excel(r"C:\Somish\plotx\Anshul_PlotX3 LP3V1.xlsx", sheet_name='Phi Values', names=['vop','move_sd'])
  
    # bot_address = get_botaddress()
    bot_address = ['0xf71411c4198aee89a9c5bcf82cd5def94af5a3b8',
                    '0xc19a55db61c95009d1b160d6e3e2a143b2a19cd6',
                    '0x6f94c97f68d2574f6a32c37b9d165f41a5832f57',
                    '0x995237b7958043abd611d707e7073b04899ee000',
                    '0x2ed6d815eceb0e9747e5238027b5083add0e026a']
    
    market_df = get_marketdata(market_ids)
    market_df['bot'] = market_df['playerAddress'].apply(lambda x : 'Bot' if x in bot_address and x!=mc_address else '')
    market_df['bot'] = market_df.apply(lambda x : 'MC' if x['playerAddress']==mc_address else x['bot'], axis=1)
    market_df['bot'] = market_df.apply(lambda x : 'Player' if x['bot']=='' else x['bot'], axis=1)
    market_df = pd.merge(market_df, strike_price[['marketIndex', 'startTime', 'settleTime', 'assetType']], on='marketIndex', how='left')
    
    stats_start = datetime.datetime(year=2022, month=1, day=1)
    stats = stats_8_hours(stats_start)
    
    ohlc=pd.DataFrame()
    close_start = stats_start.replace(tzinfo=None)
    close_start = int((close_start - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    close_end = datetime.datetime(2022, 2, 5).replace(tzinfo=None)
    close_end = int((close_end - datetime.datetime(1970, 1, 1)).total_seconds()*1000)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']:
        ohlc1 = get_ohlc(symbol, close_start, close_end, '1m')
        ohlc1['datetime'] = ohlc1['datetime'].apply(pd.to_datetime)
        ohlc = ohlc.append(ohlc1, ignore_index=True)
    
    output_dir = r"C:\Somish\plotx\mc_tsm\mc_bot_algo_20_25_25_30"
    
    for t_l in total_liquidity_list:
        for r_l in remove_bot_list:
            if r_l==True:
                r_l_p=False
            else:
                r_l_p=True
            d, df = incremental_update(t_l, r_l)
            df.to_csv(r"{}\raw_botadjusted_{}_pr4_2cent_percentsplit_{}_{}.csv".format(output_dir, r_l_p, mc_multiplier, t_l), index=False)
            df_pnl = pd.DataFrame([d]).T.reset_index()
            df_pnl.columns=['marketIndex','pnl_bots_{}_pr4_2cent_percentsplit_{}_{}'.format(r_l_p, mc_multiplier, t_l)]
            df_pnl.to_csv(r"{}\pnl_botadjusted_{}_pr4_2cent_percentsplit_{}_{}.csv".format(output_dir, r_l_p, mc_multiplier, t_l), index=False)
            
    # d, df = get_default()
    # df.to_csv(r"{}\current_raw_activitybots_as_players_pr4{}.csv".format(output_dir,mc_multiplier), index=False)
    # df_pnl = pd.DataFrame([d]).T.reset_index()
    # df_pnl.columns=['marketIndex','pnl']
    # df_pnl.to_csv(r"{}\current_pnl_activitybots_as_players_pr4{}.csv".format(output_dir,mc_multiplier), index=False)
    