# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 16:19:39 2022

@author: tusha
"""

import pandas as pd
import json
import psycopg2
import psycopg2.extras

def db_conn():
    t_host = "15.207.18.6"
    t_port = "5432"
    t_dbname = "plotx"
    t_user = "ubuntu"
    t_pw = "root"
    conn = psycopg2.connect(host=t_host, port=t_port, dbname=t_dbname, user=t_user, password=t_pw)
    return conn

def insert_to_db(df):
    """
    Using psycopg2.extras.execute_batch() to insert the dataframe fast
    """
    table = 'spinwheel_detail_new'
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

def getCurrentSize():
    conn = db_conn()
    c = conn.cursor()
    c.execute("select count(*) from spinwheel_detail")
    
    result = c.fetchall()
    count = result[0][0]
    
    c.close()
    conn.close()
    return count

def getNewSize():
    conn = db_conn()
    c = conn.cursor()
    c.execute("select count(*) from spinwheel_detail_new")
    
    result = c.fetchall()
    count = result[0][0]
    
    c.close()
    conn.close()
    return count

def createStagingTable():
    conn = db_conn()
    c = conn.cursor()
    c.execute(""" CREATE TABLE public.spinwheel_detail_new
                (
                    id character varying COLLATE pg_catalog."default" NOT NULL,
                    playeraddress character varying COLLATE pg_catalog."default" NOT NULL,
                    "timestamp" integer NOT NULL,
                    type character varying COLLATE pg_catalog."default" NOT NULL,
                    claimedid character varying COLLATE pg_catalog."default" NOT NULL,
                    claimedvalue character varying COLLATE pg_catalog."default" NOT NULL,
                    additionalspin character varying COLLATE pg_catalog."default" NOT NULL,
                    offer character varying COLLATE pg_catalog."default",
                    expiretimestamp bigint,
                    claimedamount integer
                )
                WITH (
                    OIDS = FALSE
                )
                TABLESPACE pg_default;
                
                ALTER TABLE public.spinner_wheel_collection
                    OWNER to ubuntu; """)
    conn.commit()
    c.close()
    conn.close()
    return True

def drop_existing_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS spinwheel_detail")  
    conn.commit()
    c.close()
    conn.close()
    return True
    
def rename_staging_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("ALTER TABLE spinwheel_detail_new RENAME TO spinwheel_detail")
    conn.commit()
    c.close()
    conn.close()
    return True

def drop_staging_table():
    conn = db_conn()
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS spinwheel_detail_new")  
    conn.commit()
    c.close()
    conn.close()
    return True    
    
file = open(r"C:\Somish\plotx\bplot dump\spinner_wheel_collection.csv", 'r')
data = file.read()
file.close()
data = data.replace("ObjectId(", "")
data = data.replace("ISODate(", "")
data = data.replace(")", "")

data1 = [e+"}" for e in data.split("}\n") if e]
data1[-1] = data1[-1].replace("}}","}")
data = [json.loads(i) for i in data1]

df = pd.DataFrame(data)
df.columns = [i.replace("_","").lower() for i in df.columns]

df['id'] = df['id'].apply(lambda x : list(x.values())[0])
df = df.astype(object)
df = df.where(pd.notnull(df), None)
createStagingTable()
insert_to_db(df)

if getNewSize()>getCurrentSize():
    print("Audit Success")
    drop_existing_table()
    rename_staging_table()
else:
    drop_staging_table()
    