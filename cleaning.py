# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 20:40:22 2019

@author: enigmarikki
"""
import pandas as pd
df_test_out_en = pd.read_csv('outputtest_en.csv')
df_test_out_fr = pd.read_csv('outputtest_fr.csv')
df_test_out_es = pd.read_csv('outputtest_es.csv')

df_test_en = pd.read_csv('test_en.csv')
df_test_fr = pd.read_csv('test_fr.csv') 

df_test_es = pd.read_csv('test_es.csv')

posen = df_test_en['pos']
posfr = df_test_fr['pos']
poses = df_test_es['pos']

dic = {0:'Closed' , 1:'Closed with explanation' , 2:'Closed with monetary relief' , 3:'Closed with non-monetary relief', 4:'Untimely response'}

te = []
cleanout = []
for i in range(len(posen)+len(posfr)+len(poses)):    
    for j in range(len(posen)):
        if i==posen[j]:
            te.append('Te-'+str(i+1))
            cleanout.append(dic[df_test_out_en['output'][j]])
            print(i,' done...')
            break
    for j in range(len(posfr)):
        if i==posfr[j]:
            te.append('Te-'+str(i+1))
            cleanout.append(dic[df_test_out_fr['output'][j]])
            print(i,' done...')
            break
    for j in range(len(poses)):
        if i==poses[j]:
            te.append('Te-'+str(i+1))
            cleanout.append(dic[df_test_out_es['output'][j]])
            print(i,' done...')
            break
        
sub = pd.DataFrame()
sub['Complaint-ID'] = te
sub['Complaint-Status'] = cleanout
sub.to_csv('sub.csv')