# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:38:17 2022

@author: tusha
"""

import pandas as pd
import math 

def vop4_pos(x, option, e1, e2, e3):
    try:
        if option==1:
            log_value = math.log(1 + (x/e1))
            return x + ((e2+e3) * log_value)
        elif option==2:
            log_value = math.log(1 + (x/e2))
            return x + ((e1+e3) * log_value)
        elif option==3:
            log_value = math.log(1 + (x/e3))
            return x + ((e1+e2) * log_value)
    except ZeroDivisionError:
        return 0
    
df = pd.read_excel(r"C:\Somish\plotx\pr4_testcase_3bucket.xlsx")
# df['fees'] = df.apply(lambda x : 0 if x['bot']=='MC' else x['participationValue']*0.02, axis=1)
df['fees'] = df['participationValue']*0.02
df['actualParticipationValue'] = df['participationValue']-df['fees']

df['option1'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==1 else 0, axis=1) 
df['option2'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==2 else 0, axis=1) 
df['option3'] = df.apply(lambda x : x['actualParticipationValue'] if x['optionNumber']==3 else 0, axis=1) 

df['cumm_1'] = df['option1'].cumsum().shift(1).fillna(0)
df['cumm_2'] = df['option2'].cumsum().shift(1).fillna(0)
df['cumm_3'] = df['option3'].cumsum().shift(1).fillna(0)

df['new_pos'] = df.apply(lambda x : vop4_pos(x['actualParticipationValue'], x['optionNumber'],x['cumm_1'], x['cumm_2'], x['cumm_3']), axis=1)

df['vop'] = df['actualParticipationValue']/df['new_pos']
df['vop'] = df['vop']+0.02
df['vop'] = df['vop'].apply(lambda x : x if x<0.98 else 0.98)
df['vop'] = df.apply(lambda x : 0.33 if x['bot']=='MC' else x['vop'], axis=1)
df['new_pos'] = df['actualParticipationValue']/df['vop']
# df.to_csv(r"C:\Somish\plotx\pr4_testcase_3bucket_result.csv", index=False)
