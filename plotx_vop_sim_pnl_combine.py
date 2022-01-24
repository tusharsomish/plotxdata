# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 15:59:21 2021

@author: tusha
"""

import pandas as pd

mc_multiplier=1.1

output_dir = "C:\Somish\plotx\simulation1"


df1 = pd.read_csv(r"{}\current_pnl_{}.csv".format(output_dir,mc_multiplier))
df1.columns=['marketIndex', 'pnl_current']

df2 = pd.read_csv(r"{}\algo_pnl_{}.csv".format(output_dir,mc_multiplier))
df2.columns=['marketIndex', 'pnl_algo']

df3 = pd.read_csv(r"{}\neutral_pnl_{}.csv".format(output_dir,mc_multiplier))
df3.columns=['marketIndex', 'pnl_neutral']

df4 = pd.read_csv(r"{}\voptsm_pnl_{}.csv".format(output_dir,mc_multiplier))
df4.columns=['marketIndex', 'pnl_voptsm']

df5 = pd.read_csv(r"{}\voptsm_pnl_1_{}.csv".format(output_dir,mc_multiplier))
df5.columns=['marketIndex', 'pnl_voptsm_1']

df6 = pd.read_csv(r"{}\voppr4_pnl_{}.csv".format(output_dir,mc_multiplier))
df6.columns=['marketIndex', 'pnl_voppr4']

df7 = pd.read_csv(r"{}\vophybrid_pnl_{}.csv".format(output_dir,mc_multiplier))
df7.columns=['marketIndex', 'pnl_vophybrid']



df = pd.concat([df1, df2, df3, df4, df5, df6, df7], join='inner', axis=1)
df = df.loc[:,~df.columns.duplicated()]
df.to_csv(r"{}\pnl_combined2cent_{}.csv".format(output_dir,mc_multiplier), index=False)
# df1 = df[df['marketIndex']==54635]


df1 = pd.read_csv(r"{}\voptsm_raw_{}.csv".format(output_dir,mc_multiplier))
df1 = df1[['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop','new_pos', 'reward']]
df1.columns=['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop_tsm','new_pos_tsm', 'reward_tsm']

df2 = pd.read_csv(r"{}\voptsm_raw_1_{}.csv".format(output_dir,mc_multiplier))
df2 = df2[['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop','new_pos', 'reward']]
df2.columns=['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop_tsm_1','new_pos_tsm_1', 'reward_tsm_1']

df3 = pd.read_csv(r"{}\voppr4_raw_{}.csv".format(output_dir,mc_multiplier))
df3 = df3[['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop','new_pos', 'reward']]
df3.columns=['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop_pr4','new_pos_pr4', 'reward_pr4']

df4 = pd.read_csv(r"{}\vophybrid_raw_{}.csv".format(output_dir,mc_multiplier))
df4 = df4[['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop','new_pos', 'reward']]
df4.columns=['marketIndex', 'playerAddress', 'participationValue', 'optionNumber', 'position', 'vop_hybrid','new_pos_hybrid', 'reward_hybrid']

df = pd.concat([df1, df2, df3, df4], join='inner', axis=1)  
df = df.loc[:,~df.columns.duplicated()]
df.to_csv(r"{}\raw_combined2cent_{}.csv".format(output_dir,mc_multiplier), index=False)

