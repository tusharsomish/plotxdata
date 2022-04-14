# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 19:34:09 2021

@author: tusha
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import requests
from requests.auth import HTTPBasicAuth
import datetime
import time
import json
import psycopg2
import psycopg2.extras

pd.options.mode.chained_assignment = None

def db_conn():
    t_host = "15.206.148.131" # KEEP ME UPDATED
    t_port = "5432"
    t_dbname = "plotx"
    t_user = "ubuntu"
    t_pw = "root"
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    return conn

def latest_date():
    conn = db_conn()
    c = conn.cursor()

    c.execute("""select max(time)
              from mixpanel_raw_data""")    
    result = c.fetchall()
    result = result[0][0]
    
    time_str = datetime.datetime.utcfromtimestamp(result) + datetime.timedelta(days=1)
    time_str = time_str.strftime('%Y-%m-%d')
    
    c.close()
    conn.close()
    return time_str

def existing_colnames():
    conn = db_conn()
    c = conn.cursor()

    c.execute("""select * from mixpanel_raw_data where false""")    
    # result = c.fetchall()
    result = [desc[0] for desc in c.description]
    c.close()
    conn.close()
    return result
    
def insert_to_db(df, table):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe fast
    """
    conn = db_conn()
    c = conn.cursor()
    
    #table = 'raw_data_dump'

    try:
        if len(df) > 0:
            print("Insert to DB Begin")
            df_columns = list(df)
            # create (col1,col2,...)
            df_columns = ['"'+i+'"' for i in df_columns]
            columns = ",".join(df_columns)
        
            # create VALUES('%s', '%s",...) one '%s' per column
            values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 
        
            #create INSERT INTO table (columns) VALUES('%s',...)
            insert_stmt = """INSERT INTO {} ({}) {}""".format(table,columns,values)
        
            psycopg2.extras.execute_batch(c, insert_stmt, df.values)
            conn.commit()
            c.close()
            conn.close()
            print(f'Insert into {table} complete')
        return True
    except Exception as e:
        print(e)
        raise
    finally:
        c.close() 
        conn.close()

def add_new_cols(old_names, new_names):
    
    conn = db_conn()
    c = conn.cursor()
    
    for idx, col in new_names.iterrows():
        colname = col['colname']
        datatype = col['dtype']
        if colname not in old_names:
            print(f"Adding new column: {colname} to mixpanel_raw_data")
            if 'int' in datatype or 'float' in datatype:
                print(f'{colname} real')
                # c.execute(f"ALTER TABLE mixpanel_raw_data ADD COLUMN {colname} real ")
            else:
                print(f'{colname} char')
                # c.execute(f"ALTER TABLE mixpanel_raw_data ADD COLUMN {colname} VARCHAR ")
    conn.commit()
    c.close()
    conn.close()
    return
    
def get_mixpanel_data():
    start_date = latest_date() 
    
    end_date = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    # end_date = '2022-04-09'  #UPDATE ME  
    
    print(f"Fetching data from Mixpanel from {start_date} to {end_date}")
    
    url = f"https://data.mixpanel.com/api/2.0/export?from_date={start_date}&to_date={end_date}"  
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Basic OTBiMGU4NjMxMDM3ZjhlNTQ3ZDJmOWQ4Y2NjNTc2N2I6'
        }
    
    req = requests.get(url, headers=headers)
    
    req1 = req.text.split('\n')
    req1 = [json.loads(i) for i in req1 if len(i)>0]
    
    df = pd.DataFrame(req1)
    df = pd.concat([df, df['properties'].apply(pd.Series)], axis=1)
    df = df.drop('properties', axis=1)
    print("Data Fetch successful")
    return df


if __name__ == '__main__':
    
    df = get_mixpanel_data()    
    
    new_colnames = df.dtypes.astype(str).reset_index()
    new_colnames.columns=['colname','dtype']
    new_colnames['colname'] = new_colnames['colname'].str.replace("$","").str.replace(" ","_").str.lower()
    
    df.columns = new_colnames['colname'].to_list()
    df = df.loc[:,~df.columns.duplicated()]
    
    if 'geodata' in df.columns:
        df1 = df['geodata']
        df1 = df1.apply(pd.Series)
        df1 = df1.drop(0, axis=1)
        cols = ['geodata.'+i for i in df1.columns]
        df1.columns = cols
        
        df2 = df1['geodata.time_zone'].apply(pd.Series)
        df2 = df2.drop(0, axis=1)
        cols = ['geodata.time_zone.'+i for i in df2.columns]
        df2.columns = cols
        
        df3 = df1['geodata.currency'].apply(pd.Series)
        df3 = df3.drop(0, axis=1)
        cols = ['geodata.currency.'+i for i in df3.columns]
        df3.columns = cols
        
        geo_df = pd.concat([df1,df2,df3], axis=1)
        geo_df = geo_df.drop(['geodata.time_zone', 'geodata.currency'], axis=1, errors='ignore')
        
        df = df.drop(['geodata'], axis=1, errors='ignore')
        df = pd.concat([df,geo_df], axis=1)
    
    old_colnames = existing_colnames()
    new_colnames = df.dtypes.astype(str).reset_index()
    new_colnames.columns=['colname','dtype']
    new_colnames['colname'] = new_colnames['colname'].str.replace("$","").str.replace(" ","_").str.lower()
    df.columns = new_colnames['colname'].to_list()
    
    add_new_cols(old_colnames, new_colnames) # CHECK ME
    
    df['coupon_redeem_date'] = df['coupon_redeem_date'].astype(str)
    df['coupon_redeem_date'] = df['coupon_redeem_date'].replace("{}", np.nan)
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)

    insert_to_db(df, 'mixpanel_raw_data')
 
    
 
# df1 = df.copy()
# df1 = df[~df['coupon_redeem_status'].isna()]
# df1['coupon_redeem_date'] = df['coupon_redeem_date'].astype(str)
# df1['coupon_redeem_date'] = df1['coupon_redeem_date'].str.replace("{}",'None')
