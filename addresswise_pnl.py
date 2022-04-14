# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:55:46 2022

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


def get_marketdata(start_time, end_time):
    conn = db_conn()
    c = conn.cursor()
    c.execute(f""" SELECT * FROM market_predictions 
              WHERE db_timestamp=253402257599 AND status='Settled'
              AND "transactionTime" BETWEEN '{start_time}' and '{end_time}'
              AND "marketIndex" in (select distinct "marketIndex"
                                    from market_predictions 
                                    where "playerAddress"='0x946609c05cde5b66e47af867d6a09d8ac4e0de6e') """)
    tuples = c.fetchall()
    colnames = [desc[0] for desc in c.description]
    conn.close()
    df = pd.DataFrame(tuples, columns=colnames)
    return df


mc_address = '0x946609c05cde5b66e47af867d6a09d8ac4e0de6e'

prediction_bot = ['0x3943a2a84d53a3b7bd2c66148384c2c35f7d2e7f',
                  '0xcf3c0438e4de0f9f759e83d359d91100036a0ae8',
                  '0xdf80e17b24150ca74084da954ec9bd11787ec40a',
                  '0xbff529390c9ee673460a2cb32b5a5118d33ffca7',
                  '0x84b952fd8c7556081ee3642c933e06d7383bd3b6',
                  '0x602f0507161fc8062cdb79e322c4fec55c46a8ff',
                  '0xefd15b39fed1112b5d0aefd286578706a4a100e2',
                  '0x100be3e58c12bb392f284aec635daadb939be72d']

algo_bot = ['0x1c56dc20ce48b16614b321aadaafd0a0ca41f841',
            '0xef618401aa266067e6a1e6b289d1a5a17c1cd45f',
            '0x0fb7f9129eab9e4887d199f960f643163b3e2fe4',
            '0xe9809195fa27ed06d7ca92c7a2b26258a53dcf93',
            '0x8650f27a0eb488b89e492a31ed3a9151dab30ad7']

df = get_marketdata('2022-03-24', '2022-03-25')
df['bot']=''
df['bot'] = df.apply(lambda x : 'prediction_bot' if x['playerAddress'] in prediction_bot else x['bot'], axis=1)
df['bot'] = df.apply(lambda x : 'algo_bot' if x['playerAddress'] in algo_bot else x['bot'], axis=1)
df['bot'] = df.apply(lambda x : 'mc' if x['playerAddress']==mc_address else x['bot'], axis=1)
df['bot'] = df.apply(lambda x : 'player' if (x['playerAddress'] not in prediction_bot) and (x['playerAddress'] not in algo_bot) and (x['playerAddress']!=mc_address)  else x['bot'], axis=1)
df['participationValue'] = df['participationValue']/100000000
df['returnInPlot'] = df['returnInPlot']/100000000


total={}
for mid in list(dict.fromkeys(df['marketIndex'])):
    df1 = df[df['marketIndex']==mid]
    
    prediction_bot_count = df1.groupby('playerAddress')['id'].count().reset_index()
    prediction_bot_count = prediction_bot_count[prediction_bot_count['playerAddress'].isin(prediction_bot)]
    prediction_bot_count = prediction_bot_count.rename(columns={'id':'prediction_bot_count'})
    
    if len(prediction_bot_count)>0:
        prediction_bot_count = prediction_bot_count['prediction_bot_count'].iloc[0]
    else:
        prediction_bot_count=0
    
    pred_bot = list(dict.fromkeys(df1['playerAddress']))
    unique_pred_bot = list(set(pred_bot) & set(prediction_bot))
    unique_pred_bot = len(unique_pred_bot)
    
    # df1 = pd.merge(df1, prediction_bot_count, on='playerAddress', how='left')
    
    winning_option = df1['optionNumber'][df1['returnInPlot']>0].iloc[0]
    
    df1['fees'] = df1.apply(lambda x : 0 if x['bot'] in ['mc', 'prediction_bot'] else x['participationValue']*0.02, axis=1)
    df1['mc_share_of_fees'] = df1['fees']*0.4
    df1['actualParticipationValue'] = df1['participationValue']-df1['fees']
    df1['contributionPool'] = df1.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']!=winning_option else 0, axis=1)
    
    total_position = df1['position'][df1['optionNumber']==winning_option].sum()
    total_contribution = df1['contributionPool'].sum()
    mc_share_from_contribution_pool = total_contribution*0.05
    actual_contribution = total_contribution - mc_share_from_contribution_pool
    mc_share_of_fees = df1['mc_share_of_fees'].sum()
    
    df1['reward'] = df1.apply(lambda x : (x['actualParticipationValue']+(x['position']/total_position)*actual_contribution) if x['optionNumber']==winning_option else 0, axis=1)
    
    bot_mc_return = df1['reward'][df1['bot'].isin(['mc'])].sum()
    in_pnl = df1['actualParticipationValue'][df1['bot'].isin(['mc'])].sum()  
    out_pnl = mc_share_from_contribution_pool + mc_share_of_fees + bot_mc_return
    
    pnl = out_pnl - in_pnl
   
    total.update({mid:[pnl,unique_pred_bot,prediction_bot_count]})


total_df = pd.DataFrame([total]).T
total_df['pnl'] = total_df[0].str[0]
total_df['unique_prediction_bot'] = total_df[0].str[1]
total_df['prediction_bot_count'] = total_df[0].str[2]
total_df = total_df.reset_index()
total_df = total_df.drop(columns=[0])
total_df.to_csv(r"C:\Somish\plotx\24_25_market_pnl.csv",index=False)

# total_in = df['participationValue'][df['playerAddress']==mc_address].sum()
# total_out = df['returnInPlot'][df['playerAddress']==mc_address].sum()
# total_out - total_in

    


