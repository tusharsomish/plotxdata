# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:21:53 2021

@author: tusha
"""

import pandas as pd
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
        

file = open(r"C:\Somish\plotx\addresses_predictionBot_v4.json", 'r')
data = file.read()
file.close()
l = json.loads(data)
df = pd.DataFrame(l, columns=['bot_address'])
insert_to_db(df, 'bot_address')
