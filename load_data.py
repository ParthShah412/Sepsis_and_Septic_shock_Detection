# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:02:17 2019

@author: pvsha
"""

import pandas as pd
import numpy as np
import pickle as pk
import dask.dataframe as dd
from datetime import datetime
from matplotlib.plt import pyplot

def load_df(filename,nrows=10000000): 
    return pd.read_csv(filename,sep=',',skip_blank_lines=False,low_memory=False,nrows=nrows)

def return_item_val(item_df, item_id):
    return item_df[item_df['ITEMID']==item_id]['LABEL'].values[0]

def get_date(date_str):
    return datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')

#for the time being everything is converted to minutes as the given data is in minutes
def get_date_range(date1, date2):
    if(date2>date1):
        diff = date2 - date1
    else:
        diff = date1 - date2
    return diff.days*1440 + int(diff.seconds/60)

#for the time-being we are fixing the chart-time as time measure. We can also have store-time here
def get_date_item_vec(chart_events,subject_id,item_id, chart_time=True):
    if(chart_time):
        vec = chart_events[(chart_events['SUBJECT_ID']==subject_id)&(chart_events['ITEMID']==item_id)]\
        [['VALUE','CHARTTIME']].values
        
    else:
        vec = chart_events[(chart_events['SUBJECT_ID']==subject_id)&(chart_events['ITEMID']==item_id)]\
        [['VALUE','STORETIME']].values
    item_vec = vec[:,0]
    time_vec = vec[:,1]
    
    return item_vec, time_vec

#This determines the total sample we should have.
def get_time_interval(time_vec):
    time_vec_list = time_vec.sort()
    minute_info = get_date_range(time_vec_list[0],time_vec_list[-1])
    return minute_info

if __name__=='__main__':
    chart_events = load_df('./Data/CHARTEVENTS.csv')
    item_df = load_df('./Data/D_ITEMS.csv')
    print(chart_events.columns)
    print(item_df.columns)