# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 12:57:01 2019

@author: Administrator
"""
import os
os.chdir(r'D:\Reject_Inference_project\rmpgy')
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import json


##(1)读取数据
train_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\train_clear_month.csv',\
                         header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
valid_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\valid_data_public.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
test_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\test_data_private.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
print('Train data:',train_data.shape,'Test data:',test_data.shape,'Valid data:',valid_data.shape)

##(2)读取json文件
with open(r'D:\Reject_Inference_project\output\var_type_dict.json','r') as f:
    var_type_dict = json.loads(f.readline())

##(3)临时数据转换工作
index_var = [var_type_dict.get('source_index'),var_type_dict.get('time_index'),var_type_dict.get('sample_index')]
numeric_var = var_type_dict.get('numeric_var',[])
category_var = ['os_type','province_type','address_type','gender']
y_label_var = var_type_dict.get('y_label_var','user_type')
print('数值型：',len(numeric_var),'类别型',len(category_var),'标签:',len(y_label_var))
train_data[var_type_dict.get('category_var',[])] = train_data[var_type_dict.get('category_var',[])]\
                                                    .replace({'是':1,'否':0,'Y':1,'N':0,'全国':0})

##(4)特征重要度筛选
preprocess_data_dict = {
        'train':{'X':train_data[numeric_var+category_var],'y':train_data[y_label_var]}
                        }

from pgy_model import gbdt_mix_feature_importance
from sklearn.model_selection import LeaveOneGroupOut,StratifiedKFold
from lightgbm import LGBMClassifier

## (5)按时间分组
#time_index = train_data['risk_time'].astype(np.datetime64).apply(lambda x:x.week)
#gbdt_importance_score_detail = gbdt_mix_feature_importance(data_dict=preprocess_data_dict, \
#                                                           gbdt_estimator_map={'gbm_3_0.1_500':\
#                                                            LGBMClassifier(**{'max_depth':3,'learning_rate':0.1,\
#                                                                              'subsample':0.8, 'colsample_bytree':0.8, \
#                                                                              'n_estimators':500,'n_jobs':-1})}, \
#                                                           importance_type_list=['gain'], \
#                                                           category_feature=category_var, \
#                                                           cv=LeaveOneGroupOut(), \
#                                                           groups=time_index, \
#                                                           use_validation=True)

## (5)传统的5-折StratifiedCV
gbm_model = LGBMClassifier(**{'max_depth':3,'learning_rate':0.1, 'n_estimators':500,'n_jobs':-1})
gbdt_importance_score_detail = gbdt_mix_feature_importance(data_dict=preprocess_data_dict, \
                                                           gbdt_estimator_map={'gbm_3_0.1_500':gbm_model}, \
                                                           importance_type_list=['gain'], \
                                                           category_feature=category_var, \
                                                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10), \
                                                           groups=None, \
                                                           use_validation=True)


##gain下的分数
gbdt_importance_score = pd.DataFrame(gbdt_importance_score_detail['gain']['gbm_3_0.1_500'])
##gain下的rank
gbdt_importance_rank = gbdt_importance_score.apply(lambda x:x.rank(ascending=False,method='max'),axis=0)  

gbdt_importance_score['max'] = gbdt_importance_score[gbdt_importance_score.columns[1:]].apply(lambda x:np.max(x),axis=1)
gbdt_importance_score['min'] = gbdt_importance_score[gbdt_importance_score.columns[1:]].apply(lambda x:np.min(x),axis=1)
gbdt_importance_score['avg'] = gbdt_importance_score[gbdt_importance_score.columns[1:]].apply(lambda x:np.mean(x),axis=1)
gbdt_importance_score['max-min'] = gbdt_importance_score['max']-gbdt_importance_score['min']
gbdt_importance_score['std'] = gbdt_importance_score.apply(lambda x:np.std(x),axis=1)

##存储
gbdt_importance_score.to_csv(r'D:\Reject_Inference_project\output\gbdt_importance_score_group_cv.csv')
gbdt_importance_rank.to_csv(r'D:\Reject_Inference_project\output\gbdt_importance_rank_group_cv.csv')
  

## 如果需要参照IV与PSI可以加入

## (6)特征池筛选
from pgy_model import gbdt_feature_selector
gbdt_estimator = LGBMClassifier(**{'max_depth':3,'learning_rate':0.1,'subsample':0.8, \
                                   'colsample_bytree':0.8,'reg_alpha':1,'n_estimators':20000, 'n_jobs':-1})
feature_rank = list(gbdt_importance_score.sort_values('train',ascending=False).index[:400])
##开始筛选
feature_selected,step_detail,step_outer_valid_statistic = gbdt_feature_selector(data_dict=preprocess_data_dict,\
                                                                                  gbdt_estimator=gbdt_estimator,\
                                                                                  feature_rank=feature_rank, \
                                                                                  category_features=category_var, \
                                                                                  cv_list=[StratifiedKFold(n_splits=5, shuffle=True, random_state=0)],\
                                                                                  groups_list=[None],\
                                                                                  weights_list=[1],\
                                                                                  rounds=300,\
                                                                                  step=1,\
                                                                                  auc_diff_threshold=0,\
                                                                                  auc_initial=0.5)
category_feature_selected = [i for i in feature_selected if i in category_var]
numeric_feature_selected = [i for i in feature_selected if i not in category_var]

feature_dict = {}
feature_dict['numeric_var'] = numeric_feature_selected
feature_dict['category_var'] = category_feature_selected
## 入选的特征
with open(r'D:\Reject_Inference_project\output\feature_select.json','w') as f:
    f.write(json.dumps(feature_dict))
    
## 细则存储结果
pd.DataFrame(step_outer_valid_statistic).T.to_csv(r'D:\Reject_Inference_project\output\step_outer_valid_statistic.csv',index=True)

