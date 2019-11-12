# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 00:44:01 2019

@author: HEINZ
"""

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm

def read_csv(filename, date_columns):
    return pd.read_csv(filename, sep=',', skip_blank_lines = False, parse_dates=date_columns)

def get_average(df, compare_column):
    avg_time = 0
    k = 0
    for i in range(df.loc[:,'intime'].shape[0]):
        t_susp = df.loc[i,compare_column]
        t_intime = df.loc[i,'intime']
        if(not pd.isnull(t_susp)):
            if(t_susp>t_intime):
                avg_time += (t_susp-t_intime).days + (t_susp-t_intime).seconds/(60*60*24)
                k += 1
    return avg_time/k

def generate_bell_curve(df):
    intime_vec = df['intime']
    outtime_vec = df['outtime']
    
    mean = []
    for i in range(intime_vec.shape[0]):
        if(not pd.isnull(intime_vec[i]) and not pd.isnull(outtime_vec[i])):
            mean.append((outtime_vec[i]-intime_vec[i]).days + (outtime_vec[i]-intime_vec[i]).\
                        seconds/(60*60*24))

    mean_arr = np.array(mean)
    mu = np.mean(mean_arr)
    sigma = np.std(mean_arr)
    
    x = np.linspace(np.min(mean_arr), np.max(mean_arr), intime_vec.shape[0])
    
    print(mu, sigma)
    
    plt.plot(x, norm.pdf(x, mu, sigma))
    plt.plot()
    
    return mean_arr



def change_in_sofa_after_day1(df):
    icustay_ids = df['icustay_id'].unique()    
    m = []
    k= []
    n = []
    p = []        
    for i in range(icustay_ids.shape[0]):
        temp_df = df[df['icustay_id']==icustay_ids[i]].reset_index(drop=True)
        for j in range(temp_df.shape[0]):
            print(icustay_ids[i])
            if(j>=96 and j!=temp_df.shape[0]-1):
                if(temp_df.loc[j+1,'sofa_24hours']>=temp_df.loc[j,'sofa_24hours']+2):
                    p.append(icustay_ids[i])
                    break            
            if(j>=72 and j!=temp_df.shape[0]-1):
                if(temp_df.loc[j+1,'sofa_24hours']>=temp_df.loc[j,'sofa_24hours']+2):
                    n.append(icustay_ids[i])
                    break            
            if(j>=48 and j!=temp_df.shape[0]-1):
                if(temp_df.loc[j+1,'sofa_24hours']>=temp_df.loc[j,'sofa_24hours']+2):
                    k.append(icustay_ids[i])
                    break
            if(j>=24 and j!=temp_df.shape[0]-1):
                if(temp_df.loc[j+1,'sofa_24hours']>=temp_df.loc[j,'sofa_24hours']+2):
                    m.append(icustay_ids[i])
                    break
    return m, k, n, p

def get_bar_plot(mean_arr):
    groups = ['<5 days','<15 days','<25 days','>25 days']
    major_list = [0]*4
    for i in mean_arr:
        if(i>25):
            major_list[3] = major_list[3] + 1
        elif(i>15):
            major_list[2] = major_list[2] + 1
        elif(i>5):
            major_list[1] = major_list[1] + 1
        else:
            major_list[0] = major_list[0] + 1 
    
    plt.bar(groups, major_list)
    plt.ylabel('# of patients')
    plt.xlabel('Patient stay in ICU')
    plt.title('Count of patients grouped by days stay in ICU')
    plt.plot()
             

if __name__=='__main__':
    sepsis_3_df = read_csv('complete_sepsis_volume_responsiveness.csv', ['intime', 'outtime',\
                            'suspected_infection_time_poe', 'antibiotic_time_poe', 'blood_culture_time'])
    average_time_susp = get_average(sepsis_3_df,'suspected_infection_time_poe')
    average_time_anti_bio = get_average(sepsis_3_df,'antibiotic_time_poe')
    average_time_bc = get_average(sepsis_3_df,'blood_culture_time')
   
    mean_arr = generate_bell_curve(sepsis_3_df)
    
    pivot_sofa = read_csv('sofa_scores.csv',['starttime', 'endtime'])
    
    hrs_24, hrs_48, hrs_72, hrs_96 = change_in_sofa_after_day1(pivot_sofa)
    
    get_bar_plot(mean_arr)
    
    print(pivot_sofa.shape)
    
    print(sepsis_3_df.shape)