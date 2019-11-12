# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 16:35:26 2019

@author: HEINZ
"""

import pandas as pd
import numpy as np
import datetime

def load_csv(filename,date_columns):
    return pd.read_csv(filename,sep=',',skip_blank_lines=False,parse_dates=date_columns)

def get_columns(bolus_df, sepsis_df):
    icustay_id = sepsis_df['icustay_id']
    print(icustay_id.shape)
    final_list = []
    for i in range(icustay_id.shape[0]):
        t_susp = sepsis_df.loc[i,'suspected_infection_time_poe']
        bolus_temp = bolus_df[bolus_df['icustay_id']==icustay_id[i]].reset_index(drop=True)
        for j in range(bolus_temp.shape[0]):
            if(float(bolus_temp.loc[j,'crystalloid_bolus']) > 500.0):
                final_list.append([icustay_id[i],t_susp,bolus_temp.loc[j,'charttime']])
    return final_list
 
def make_df(list_data):
    np_arr = np.array(list_data)
    print(np_arr.shape)
    
    df_new = pd.DataFrame(np_arr)
    df_new.columns = ['icustay_id', 't_susp', 't_charttime']
    
    return df_new

def add_bool(df):
    unique_icu_stay = np.unique(df['icustay_id'].reset_index(drop=True))
    df['closest_to_suspicion_time'] = [False]*df.shape[0]
    k = 0
    for i in range(unique_icu_stay.shape[0]):
        print(unique_icu_stay[i])
        temp_df = df[df['icustay_id']==unique_icu_stay[i]]
        print(type(temp_df.loc[0,'t_charttime']))
        idx = 0
        min_delta = datetime.timedelta(days=100, hours=23, minutes=59,seconds=59, microseconds=999999)
        for j in range(temp_df.shape[0]):
            if(abs(temp_df.loc[j,'t_susp']-temp_df.loc[j,'t_charttime'])<min_delta):
                min_delta = abs(temp_df.loc[j,'t_susp']-temp_df.loc[j,'t_charttime'])
                idx = j + k
        k = k + temp_df.shape[0]-1
        df.loc[idx,'closest_to_suspicion_time'] = True
    return df

if __name__ == '__main__':
    
    bolus_df = load_csv('./bolus.csv',['charttime'])
    sepsis_df = load_csv('Complete_sepsis3.csv',['suspected_infection_time_poe'])
    
    final_list = get_columns(bolus_df, sepsis_df)
    
    df_new = make_df(final_list)
    
#    df = add_bool(df_new)