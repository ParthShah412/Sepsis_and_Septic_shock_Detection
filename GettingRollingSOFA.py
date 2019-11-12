# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:57:20 2019

@author: HEINZ
"""

import numpy as np
import psycopg2
import pandas as pd
import pandas.io.sql as psql
import datetime 
import Queries_modified

def get_icustay_intime_outtime(query_result):
    lst = []
    for res in query_result:
        if(res[10] is None):    
            lst.append([res[3], res[9], res[9]])
        else:
            lst.append([res[3], res[9], res[10]])
    return np.array(lst)


if __name__=='__main__':
    connection  = psycopg2.connect(host='localhost', database='mimic',\
                                   user='postgres', password='0357')
    cursor = connection.cursor()
    dataframe = cursor.execute('select * from mimiciii.icustays')
    lst = []
    for i in range(24):
        new_df = Queries_modified.get_per_hour_urine_output(cursor, i)
        lst.append(cursor.fetchall())
    
    df = pd.DataFrame(lst[0])
    
    for i in range(1, len(lst)):
        df = pd.concat(df,pd.DataFrame(lst[i]))
    print(df.shape)
    