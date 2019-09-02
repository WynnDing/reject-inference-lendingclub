# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 11:04:13 2019

@author: Administrator
"""

import os
os.chdir(r'D:\Reject_Inference_project\rmpgy')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime as dt
from pgy_dataclean import data_clean

## 一、数据标签与时间处理
config_dict = {'data_path':r'D:\Reject_Inference_project\raw_data',
               'data_list':['xm_raw_data'],
               'config_name':r'D:\Reject_Inference_project\var_config_0306.xlsx',
               'train_output_path':r'D:\Reject_Inference_project\data_clean_combine_label\train_clear_month.csv',
               'test_output_path':r'D:\Reject_Inference_project\data_clean_combine_label\test_clear_month.csv',
               'var_type_dict_output_path':r'D:\Reject_Inference_project\output\var_type_dict.json',
               'data_select':{'table_to_drop':['arc_mxdata_taobao','arc_kexin_xbehavior'],
                              'var_to_drop':['zm_score'],
                              'ext_var':['first_login_phone_type','first_login_time','fitsr_register_time','gmt_mobile','last_login_phone_type','last_login_time','nation'],
                              'time_to_stay':{'xm_raw_data_train':['2018-09-13 00:00:00','2018-11-10 23:59:59'],
                                              'xm_raw_data_test':['2018-11-11 00:00:00','2018-12-10 23:59:59'],
                                              }},
               'source_index':'source_index',
               'time_index':'risk_time',
               'sample_index':'consumer_no',
               'y_type':'type_1',
               'is_grey_good':False,
               'y_column':'user_type',
               'save_all_data':False 
               }
data_clean(config_dict=config_dict)


##二、数据准备工作（包含离线的特征衍生和数据合并）
##(1)读取清洗完的数据
train_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\train_clear_month.csv',\
                         header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
test_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\test_clear_month.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
print('Ratio Train：%s'%(train_data['user_type'].sum()/len(train_data)))
print('Ratio Test：%s'%(test_data['user_type'].sum()/len(test_data)))
print('Train shape:',train_data.shape,'Test shape:',test_data.shape)

##(2)自定义的测试集切分工作
split_time = dt.strptime('2018-11-25 00:00:00','%Y-%m-%d %H:%M:%S')
valid_data_public,test_data_private = test_data[test_data['risk_time']<=split_time],test_data[test_data['risk_time']>split_time]
print('Ratio Valid public：%s'%(valid_data_public['user_type'].sum()/len(valid_data_public)))
print('Ratio Test private：%s'%(test_data_private['user_type'].sum()/len(test_data_private)))
print('Test_public:',valid_data_public.shape,'Test_private:',test_data_private.shape)

valid_data_public.to_csv(r'D:\Reject_Inference_project\data_clean_combine_label\valid_data_public.csv',index=False)
test_data_private.to_csv(r'D:\Reject_Inference_project\data_clean_combine_label\test_data_private.csv',index=False)

