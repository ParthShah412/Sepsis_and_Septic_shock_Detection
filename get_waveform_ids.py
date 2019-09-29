# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 18:13:34 2019

@author: pvsha
"""
import pandas as pd
import numpy as np


def load_file(filename):
    f = open(filename,'r')
    f_list = f.readlines()
    return f_list

def write_file(filename, lst):
    f = open(filename,'w')
    for i in lst:
        f.write(str(i))
        f.write('\n')
    f.close()

def get_subject_ids(string_list):
    subject_id_list = []
    for i in range(len(string_list)):
        _,_,subject_id_date = string_list[i].split('/')
        subject_id,_,_,_,_,_ = subject_id_date.split('-')
        subject_id_list.append(int(subject_id.strip('p')))
    return subject_id_list

if __name__ == '__main__':
    f_list = load_file('./Data/Waveformfiles.txt')
    subject_id_list = get_subject_ids(f_list)
    write_file('./Data/SubjectIDs.txt',subject_id_list)
    print(len(f_list))
    print(len(subject_id_list))