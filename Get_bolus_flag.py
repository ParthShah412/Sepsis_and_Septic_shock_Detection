# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:38:48 2019

@author: HEINZ
"""

import numpy as np
import pandas as pd
import datetime

def load_csv(filename, list_date_columns):
    return pd.read_csv(filename,sep=',',skip_blank_lines=False, parse_dates=list_date_columns)


def get_indexes_sofa_matches_suspision(sofa_df, sepsis_df, bolus_dict):
    
    icustay_ids = sepsis_df['icustay_id']
    volume_responsive_dict = {}
    dt = None
    for i in range(icustay_ids.shape[0]):
        temp_sofa = sofa_df[sofa_df['icustay_id']==icustay_ids[i]].reset_index(drop=True)
        if(bolus_dict[icustay_ids[i]]=='NaN' or bolus_dict[icustay_ids[i]]<500):
            volume_responsive_dict[icustay_ids[i]] = 'NaN'
        else:
            
            if(pd.isnull(sepsis_df.loc[i,'suspected_infection_time_poe'])):
                dt = sepsis_df.loc[i,'intime']
            else:
                dt = sepsis_df.loc[i,'suspected_infection_time_poe']
            min_delta = datetime.timedelta(days=0, hours=23, minutes=59,\
                                           seconds=59, microseconds=999999)
            volume_val = 0
            mean_bp = 0
            for j in range(temp_sofa.shape[0]):
                if(abs(temp_sofa.loc[j,'starttime'] - dt)<min_delta):
                    min_delta = abs(temp_sofa.loc[j,'starttime']-dt)
                    if(j!=temp_sofa.shape[0]-1):
                        delta_bp = (temp_sofa.loc[j+1,'meanbp_min']/temp_sofa.loc[j,'meanbp_min'])
                        if(delta_bp>=1.15):
                            volume_val = 1
                        mean_bp = temp_sofa.loc[j,'meanbp_min']
            volume_responsive_dict[icustay_ids[i]] = [volume_val, mean_bp]
    return volume_responsive_dict

def get_patient_wise_responsive_tag(bolus_df, sepsis_df):
    icustay_ids = sepsis_df['icustay_id']
    bolus_dict = {}
    dt = None
    for i in range(icustay_ids.shape[0]):
        temp_bolus = bolus_df[bolus_df['icustay_id']==icustay_ids[i]].reset_index(drop=True)
        if(temp_bolus.shape[0]==0):
            bolus_dict[icustay_ids[i]] = 'NaN'
            continue
        if(pd.isnull(sepsis_df.loc[i,'suspected_infection_time_poe'])):
            dt = sepsis_df.loc[i,'intime']
        else:
            dt = sepsis_df.loc[i,'suspected_infection_time_poe']
        if(dt<sepsis_df.loc[i,'intime'] or dt>sepsis_df.loc[i,'outtime']):
            bolus_dict[icustay_ids[i]] = 'NaN'
        else:
            min_delta = datetime.timedelta(days=0, hours=23, minutes=59,\
                                           seconds=59, microseconds=999999)
            bolus_val = 0
            for j in range(temp_bolus.shape[0]):
#                print(i)
                if(abs(temp_bolus.loc[j,'charttime'] - dt)<min_delta):
                    min_delta = abs(temp_bolus.loc[j,'charttime'] - dt)
                    bolus_val = temp_bolus.loc[j,'crystalloid_bolus']
            bolus_dict[icustay_ids[i]] = bolus_val
    
    return bolus_dict    
                    


if __name__=='__main__':
    volume_resp_df = load_csv('complete_sepsis_volume_responsiveness.csv'\
                              ,['suspected_infection_time_poe','intime', 'outtime'])
#    sofa_df = load_csv('sofa_scores.csv', ['starttime', 'endtime'])
#    complete_sepsis_df = load_csv('Complete_sepsis3.csv',['suspected_infection_time_poe',\
#                                                          'intime', 'outtime'])
#    bolus_df = load_csv('bolus.csv',['charttime'])
#    
#    
#    bolus_dict = get_patient_wise_responsive_tag(bolus_df, complete_sepsis_df)
#    volume_dict = get_indexes_sofa_matches_suspision(sofa_df, complete_sepsis_df, bolus_dict)
#    k = 0
#    for i in volume_dict.keys():
#        complete_sepsis_df.loc[k,'Volume_responsiveness'] = volume_dict[i][0]
#        complete_sepsis_df.loc[k,'Blood_pressure'] = volume_dict[i][1]
#        k = k + 1
#    print(bolus_df.iloc[0,1])
    