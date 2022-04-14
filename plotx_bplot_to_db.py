# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:43:08 2021

@author: tusha
"""

import pandas as pd
import numpy as np
import json
import psycopg2
import psycopg2.extras

def db_conn():
    t_host = "13.127.216.173"
    t_port = "5432"
    t_dbname = "plotx"
    t_user = "ubuntu"
    t_pw = "root"
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    return conn

def sizeCurrentTable():
    conn = db_conn()
    c = conn.cursor()
    c.execute("select count(*) from user_claimed_for_airdrop_app")
    result = c.fetchall()
    count = result[0][0]
    conn.close()    
    return count

def sizeNewTable():
    conn = db_conn()
    c = conn.cursor()
    c.execute("select count(*) from user_claimed_for_airdrop_app_new")
    result = c.fetchall()
    count = result[0][0]
    conn.close()    
    return count

def insert_to_db(df):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe fast
    """
    table = 'user_claimed_for_airdrop_app_new'
    conn = db_conn()
    c = conn.cursor()   
    try:
        if len(df) > 0:
            df_columns = list(df)
            # create (col1,col2,...)
            columns = ",".join(df_columns)
        
            # create VALUES('%s', '%s",...) one '%s' per column
            values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 
        
            #create INSERT INTO table (columns) VALUES('%s',...)
            insert_stmt = """INSERT INTO {} ({}) {}""".format(table,columns,values)
        
            psycopg2.extras.execute_batch(c, insert_stmt, df.values)
            conn.commit()
            c.close()
        return True
    except Exception as e:
        print(e)
        raise
    finally:
        c.close()

def create_staging_table():
    conn = db_conn()
    c = conn.cursor()
    
    c.execute(''' DROP TABLE IF EXISTS public.user_claimed_for_airdrop_app_new;

            CREATE TABLE public.user_claimed_for_airdrop_app_new
            (
                id character varying COLLATE pg_catalog."default" NOT NULL,
                useraddress character varying COLLATE pg_catalog."default" NOT NULL,
                start_timestamp integer NOT NULL,
                end_timestamp bigint NOT NULL,
                claimedforapp double precision NOT NULL,
                latest_claimedforapp double precision NOT NULL,
                used double precision NOT NULL,
                categoryid integer NOT NULL,
                predictioncount integer NOT NULL,
                maxallocation double precision NOT NULL,
                status character varying COLLATE pg_catalog."default",
                unlocktimestamp integer
            )
            WITH (
                OIDS = FALSE
            )
            TABLESPACE pg_default;
            
            ALTER TABLE public.user_claimed_for_airdrop_app_new
                OWNER to ubuntu;''')
    conn.commit()
    c.close()
    conn.close()
    return True
    
def drop_existing_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS user_claimed_for_airdrop_app")  
    conn.commit()
    c.close()
    conn.close()
    return True
    
def rename_staging_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("ALTER TABLE user_claimed_for_airdrop_app_new RENAME TO user_claimed_for_airdrop_app")
    conn.commit()
    c.close()
    conn.close()
    return True

def drop_staging_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS user_claimed_for_airdrop_app_new")  
    conn.commit()
    c.close()
    conn.close()
    return True

text = open(r"C:\Somish\plotx\bplot dump\user_claimed_for_airdrop_app.csv", 'r')
data = text.readlines()
text.close()
data = [i.replace('{"_id":', '') for i in data]
data = [i.replace('}', '',1) for i in data]
data = [json.loads(i) for i in data]


df = pd.DataFrame(data)
df = df.rename(columns={'$oid':'id', 
                        'userAddress':'useraddress',
                        'claimedForApp':'claimedforapp',
                        'latest_claimedForApp':'latest_claimedforapp',
                        'categoryId':'categoryid', 
                        'predictionCount':'predictioncount',
                        'maxAllocation':'maxallocation',
                        'bonusRule':'bonusrule',
                        'isEligibleForBonus':'iseligibleforbonus',
                        'unlockTimestamp':'unlocktimestamp'})

df = df[['id', 'useraddress', 'start_timestamp', 'end_timestamp',
       'claimedforapp', 'latest_claimedforapp', 'used', 'categoryid', 
       'predictioncount', 'maxallocation','status', 'unlocktimestamp']]

df = df.astype(object)
df = df.where(pd.notnull(df), None)
# df['unlocktimestamp'] = df['unlocktimestamp'].astype(object).replace(np.nan, 'None')

create_staging_table()
insert_to_db(df)

if sizeNewTable()>sizeCurrentTable():
    print("Audit Success")
    drop_existing_table()
    rename_staging_table()
else:
    drop_staging_table()
