# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:18:59 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import json
import os
os.chdir(r'D:\Reject_Inference_project\rmpgy')
import warnings
warnings.filterwarnings('ignore')
import gc
from datetime import datetime as dt
from functools import wraps
import random

import lightgbm as gbm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score,classification_report
from sklearn_pandas import CategoricalImputer,DataFrameMapper
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn2pmml.decoration import ContinuousDomain,CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import gen_features
from sklearn.pipeline import FeatureUnion
from sklearn.utils import shuffle
from sklearn import metrics

from pgy_evaluation import plot_ks_curve,plot_multi_roc_curve_dict_type,\
                            plot_multi_reject_bad_curve_dict_type,\
                            plot_multi_PR_curve_dict_type,plot_density_curve
# 时间函数
def timecount():
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = dt.now()
            temp_result = func(*args, **kwargs)
            end_time = dt.now()
            time_pass = end_time-start_time
            print(("time consuming：" + str(np.round(time_pass.total_seconds()/60,2)) + "min").center(50, '='))
            return temp_result
        return wrapper
    return decorate

## 读取训练数据
def load_data():
    train_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\train_clear_month.csv',\
                         header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
    valid_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\valid_data_public.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
    test_data = pd.read_csv(r'D:\Reject_Inference_project\data_clean_combine_label\test_data_private.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
    reject_data = pd.read_csv(r'D:\Reject_Inference_project\raw_data\xm_ext_data.csv',\
                        header = 0, encoding = 'utf-8',na_values = '\\N',parse_dates=[1])
    with open(r'D:\Reject_Inference_project\output\feature_select.json','r') as f:
        feature_dict = json.loads(f.readline())
    return train_data,valid_data,test_data,reject_data,feature_dict

## 模型生成
def model_generate(param_dict,types='train'):
    lightgbm_model = gbm.LGBMClassifier().set_params(**param_dict.get('lightgbm',{}))
    xgboost_model = XGBClassifier().set_params(**param_dict.get('xgboost',{}))
    lr_model = LogisticRegression().set_params(**param_dict.get('lr',{}))
    if types == 'train':
        RF_model = RandomForestClassifier().set_params(**param_dict.get('RF',{}))
        NB_model = GaussianNB().set_params(**param_dict.get('NB',{}))
        catboost_model = CatBoostClassifier().set_params(**param_dict.get('catboost',{}))
        model_list = [lightgbm_model,xgboost_model,RF_model,NB_model,lr_model,catboost_model]
    elif types == 'eval':
        model_list = [lightgbm_model,xgboost_model,lr_model]
    else:
        raise ValueError
    return model_list

## 特征合并
def feature_union(category_feature,numeric_feature):
    mapper_category = DataFrameMapper(gen_features(
            columns = category_feature,
            classes = [CategoricalDomain,CategoricalImputer,LabelBinarizer]
            ))
    mapper_numerical = DataFrameMapper([
            (numeric_feature,[ContinuousDomain(),SimpleImputer(strategy='mean'),StandardScaler()])
            ])
    pipeline_transformer = FeatureUnion([('mapper_category',mapper_category),\
                                         ('mapper_numerical',mapper_numerical)]) 
    return pipeline_transformer


## 模型训练
def fuse_model_train(X_train,y_train,model_list,category_feature,\
                     numeric_feature,classifier_type='train'):
    pipeline_transformer = feature_union(category_feature,numeric_feature)
    if classifier_type == 'train':
        model_tuple = list(zip([str(i) for i in range(len(model_list))],model_list))
        model_fuse = VotingClassifier(estimators=model_tuple,voting='soft',\
                                      weights=[3,3,2,0.8,2.2,3])
    elif classifier_type == 'eval':
        model_tuple = list(zip([str(i) for i in range(len(model_list))],model_list))
        model_fuse = VotingClassifier(estimators=model_tuple,voting='soft',\
                                      weights=[2,2,1])
    else:
        raise ValueError
    
    pipeline_model = PMMLPipeline([
        ('mapper',pipeline_transformer),
        ('classifier',model_fuse)
    ])
    print('y_train:',y_train.shape,'ratio:',np.sum(y_train.values)/len(y_train))
    pipeline_model.fit(X_train,y_train)
    return pipeline_model

## 子模型结果输出
def submodel_evaluation(train_data,valid_data,model_list,\
                        category_feature,numeric_feature): 
    X_train = train_data[category_feature+numeric_feature] 
    y_train = train_data['user_type']
    X_valid = valid_data[category_feature+numeric_feature]
    y_valid = valid_data['user_type']
    
    pipeline_transformer = feature_union(category_feature,numeric_feature)    
    model_result_dict = {}
    for model in model_list:
        model_name = model.__class__.__name__
        print('model %s evaluation'%model_name)
        
        sub_model = PMMLPipeline([
            ('mapper',pipeline_transformer),
            ('classifier',model)
        ])
        sub_model.fit(X_train,y_train)
        predict_valid = sub_model.predict_proba(X_valid)[:,1]
        model_ks = plot_ks_curve(predict_valid,valid_data['user_type'])
        model_auc = roc_auc_score(y_valid, predict_valid)
        model_result_dict[model_name] = [model_ks,model_auc]
    return model_result_dict

## 主模型评估
def model_fuse_evaluation(model,train_data,valid_data,test_data,feature_used):
    evaluation_result = []
    X_valid = valid_data[feature_used]
    y_valid = valid_data['user_type']
    X_test = test_data[feature_used]
    y_test = test_data['user_type']
    X_train = train_data[feature_used]
    y_train = train_data['user_type']
    
    predict_valid = model.predict_proba(X_valid)[:,1]
    predict_label = model.predict(X_valid)
    valid_ks = plot_ks_curve(predict_valid,y_valid)
    valid_auc = roc_auc_score(y_valid, predict_valid)
    
    predict_test = model.predict_proba(X_test)[:,1]
    ks_test = plot_ks_curve(predict_test,y_test)
    auc_test = roc_auc_score(y_test, predict_test)  
    
    predict_train = model.predict_proba(X_train)[:,1]
    ks_train = plot_ks_curve(predict_train,y_train)
    auc_train = roc_auc_score(y_train, predict_train)    
    
    # 计算准确率
    print(classification_report(y_valid.values,predict_label,target_names=['0', '1']))
    accuracy = metrics.accuracy_score(y_valid,predict_label)
    
    # 结果保存
    evaluation_result = [valid_ks,valid_auc,ks_test,auc_test,auc_train,ks_train,accuracy]
    return evaluation_result

# 模型对拒绝样本进行预测
def reject_data_masked(reject_train,predict_reject_prob,n_sample=1000,ratio=1,bad=0.8,good=0.2):
    reject_train['user_prob'] = predict_reject_prob
    
    # 好样本与坏样本
    reject_train_temp = reject_train[((reject_train['user_prob']<1)&(reject_train['user_prob']>bad))\
                                     |((reject_train['user_prob']<good)&(reject_train['user_prob']>0))]
    
    reject_train_temp['user_type'] = reject_train_temp['user_prob'].apply(lambda x: 1.0 if x>0.5 else 0)
    print('拒绝样本:',reject_train_temp['user_type'].value_counts())
    
    # 训练样本
    n = random.randint(1,10)
    temp_1 = reject_train_temp[reject_train_temp['user_type'] == 1].sample(n_sample,random_state=n)
    temp_0 = reject_train_temp[reject_train_temp['user_type'] == 0].sample(int(ratio*n_sample),random_state=n)
    del reject_train_temp
    gc.collect()
    reject_train_combine_label = shuffle(pd.concat([temp_0,temp_1]))   
    return reject_train_combine_label

# 预测拒绝样本标签
def reject_data_predict(reject_data,pipeline_model,feature_used):
    reject_train = reject_data[feature_used]
    predict_reject_prob = pipeline_model.predict_proba(reject_train)[:,1]
    return predict_reject_prob,reject_train

# 拒绝样本与原始样本合并
def raw_combine_reject(train_data,reject_train_combine_label,feature_used):
    train_combine_reject_1 = pd.concat([train_data[feature_used+['user_type']],reject_train_combine_label\
          [feature_used+['user_type']]]).reset_index(drop=True)
    X_train_reject = train_combine_reject_1[feature_used]
    y_train_reject = train_combine_reject_1['user_type']
    return X_train_reject,y_train_reject

# 拒绝样本单轮次测试
@timecount()
def reject_main_train(reject_train,train_data,predict_reject_prob,model_list,\
                      feature_used,model_auc_raw,bad,good,diff=0):  
    reject_evalutaion_detail = {}
    pipeline_model_userful = []
    reject_train_userful = []
    for ratio in [0.3,0.5,1,2]:
        for n_sample in [1000,2000,3000,4000]:    
            print('开始 %s_%s'%(ratio,n_sample))
            # 选取拒绝标签
            reject_train_combine_label = reject_data_masked(reject_train,\
                                                            predict_reject_prob,n_sample,ratio,bad,good)
            # 合并数据训练集
            X_train_reject_1,y_train_reject_1 = raw_combine_reject(train_data,\
                                                                   reject_train_combine_label,feature_used)
            
            # 拒绝样本训练
            pipeline_model_reject_1 = fuse_model_train(X_train_reject_1,y_train_reject_1,\
                                        model_list,category_feature,numeric_feature,classifier_type='train')
            # 拒绝样本评估
            evaluation_result = model_fuse_evaluation(pipeline_model_reject_1,train_data,\
                                                      valid_data,test_data,feature_used)
            reject_evalutaion_detail[str(ratio)+'_'+str(n_sample)] = evaluation_result
            if evaluation_result[1] - model_auc_raw >= diff:
                print('reject auc:',evaluation_result[1],'raw_auc:',model_auc_raw)
                print(y_train_reject_1.value_counts())
                pipeline_model_userful.append(pipeline_model_reject_1)
                reject_train_userful.append(reject_train_combine_label)
            
    return reject_evalutaion_detail,pipeline_model_userful,reject_train_userful

# 贝叶斯调参
from pgy_model import bayes_parameter_opt_lgb
def auto_bayes_param(train_data,category_feature,numeric_feature):
    lgb_best_param = bayes_parameter_opt_lgb(train_data[category_feature+numeric_feature]\
                                             ,train_data['user_type'],category_feature)
    return lgb_best_param

# 保存
def result_save(result,name):
    writer = pd.ExcelWriter(r'D:\Reject_Inference_project\output\test_reject_time_3_drop0.8-0.2.xlsx',engine='xlsxwriter')
    result.to_excel(excel_writer=writer,sheet_name=name,index=False)
    writer.save()

# 1 读取特征
train_data,valid_data,test_data,reject_data,feature_dict = load_data()
category_feature = feature_dict['category_var']
numeric_feature = feature_dict['numeric_var']
X_train = train_data[category_feature+numeric_feature]
y_train = train_data['user_type']
X_valid = valid_data[category_feature+numeric_feature]
y_valid = valid_data['user_type']
print('X_train:',X_train.shape,'X_valid:',X_valid.shape)

reject_data_temp = reject_data[(reject_data['risk_time']>'2018-09-13 00:00:00')\
                               &(reject_data['risk_time']<'2018-11-15 00:00:00')]
#lgb_best_param = auto_bayes_param(train_data,category_feature,numeric_feature)

    
param_dict_eval = {
    'lightgbm':{
        'boosting_type': 'gbdt',
        'learning_rate':0.1, 
        'max_depth':4,
        'num_leaves':13, #28
        'reg_alpha':2.3, #2.6
        'reg_lambda':2.3, #1.3
        'n_estimators':800,# 600
        'random_state':10,
        'n_jobs':-1,
        'class_weight':{0:1,1:3},
        'subsample':0.8,
        'colsample_bytree':0.8
        },    
    'xgboost':{
         'learning_rate':0.1,
         'n_estimators':800,
         'max_depth':4,
         'objective':'binary:logistic',
         'seed':10,
         'reg_alpha':2.3,
         'reg_lambda':2.3,
         'random_state':10,
         'scale_pos_weight':3,
         'subsample':0.8,
         'colsample_bytree':0.8,
         'n_jobs':-1
        },
    'lr':{
        'C':0.5,
        'penalty':'l1',
        'random_state':10,
        'class_weight':{0:1,1:3},
        'n_jobs':-1
        },
    'RF':{
        'n_estimators':700,
        'max_depth':3,
        'random_state':10,
        'class_weight':{0:1,1:3},
        'n_jobs':-1
        },
    'NB':{
        'priors':None
        },
    'catboost':{
        'iterations':800,
        'max_depth':4,
        'learning_rate':0.1,
        'task_type':'GPU',
        'devices':[0],
        'scale_pos_weight':3,
        'random_seed':10,
        'verbose':-1
        }
    }

#model_list_base = model_generate(param_dict_train,types='eval')
model_list_eval = model_generate(param_dict_eval,types='train')

# 原始模型结果
pipeline_model_base = fuse_model_train(X_train,y_train,model_list_eval,category_feature,\
                     numeric_feature,classifier_type='train')
feature_used = category_feature + numeric_feature
# 原子模型输出
submodel_raw_result = submodel_evaluation(train_data,valid_data,model_list_eval,\
                        category_feature,numeric_feature)

# 原始模型评估
evaluation_raw = {}
evaluation_raw_details = model_fuse_evaluation(pipeline_model_base,train_data,valid_data,test_data,feature_used)
evaluation_raw['raw'] = evaluation_raw_details
model_auc_raw = evaluation_raw_details[1]
eval_df = pd.DataFrame(data = evaluation_raw,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])

# 开始第一轮迭代: 预测拒绝标签
predict_reject_prob,reject_train = reject_data_predict(reject_data_temp,pipeline_model_base,feature_used)

# 原始模型生成第一批拒绝样本测试：单轮测试
reject_evalutaion_detail,pipeline_model_userful,reject_train_userful = reject_main_train(reject_train,\
                                                train_data,predict_reject_prob,model_list_eval,feature_used,model_auc_raw,0.8,0.2,diff=0)

# 第一轮结果输出:最佳0.74537
eval_reject_df_1 = pd.DataFrame(data = reject_evalutaion_detail,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])
temp_1 = eval_reject_df_1.merge(eval_df,on=[eval_reject_df_1.index],how='inner')
result_save(temp_1,name=u'0.8_0.2_深度4_时间0913-1115_0.7452')

# 保存第一轮最佳的数据集与模型
pipeline_model_reject_1 = pipeline_model_userful[1]
userful_reject_data_1 = reject_train_userful[1]
from sklearn.externals import joblib
joblib.dump(pipeline_model_reject_1,r'D:\Reject_Inference_project\model_pkl_reject_data\best_model_1_0.7452.pkl',compress=3)
userful_reject_data_1.to_csv(r'D:\Reject_Inference_project\model_pkl_reject_data\best_data_1_0.7452.csv',index=False)


# 开始第二轮迭代: 输出最佳结果对应的model
model_auc_reject_1 = 0.7427
reject_data_temp['prob_1'] = predict_reject_prob
reject_data_temp_2 = reject_data_temp[(reject_data_temp['prob_1']<0.8)&(reject_data_temp['prob_1']>0.2)] 

predict_reject_prob_2,reject_train_2 = reject_data_predict(reject_data_temp_2,pipeline_model_reject_1,feature_used)
reject_evalutaion_detail_2,pipeline_model_userful_2,reject_train_userful_2 = reject_main_train(reject_train_2,\
                                                train_data,predict_reject_prob_2,model_list_eval,feature_used,model_auc_reject_1,0.9,0.2,diff=0)

# 第二轮结果输出:
# 阈值0.85_0.15_深度4_时间0913-1115_正则_2 0.744730 
# 阈值0.85_0.15_深度4_时间1001-1115_正则_2 0.745252
# 阈值0.85_0.15_深度4_时间1015-1115_正则_2 
eval_reject_df_2 = pd.DataFrame(data = reject_evalutaion_detail_2,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])
temp_2 = eval_reject_df_2.merge(eval_df,on=[eval_reject_df_2.index],how='inner')
result_save(temp_2,name = u'0.8_0.2_深度4_0913-1115_2轮0.7451')
# 保存这一轮的结果
pipeline_model_reject_2 = pipeline_model_userful_2[2]
userful_reject_data_2 = reject_train_userful_2[2]
from sklearn.externals import joblib
joblib.dump(pipeline_model_reject_2,r'D:\Reject_Inference_project\model_pkl_reject_data\best_model_2_0.7451.pkl',compress=3)
userful_reject_data_2.to_csv(r'D:\Reject_Inference_project\model_pkl_reject_data\best_data_2_0.7451.csv',index=False)


# 第三轮
model_auc_reject_1 = 0.7427
reject_data_temp_2['prob_2'] = predict_reject_prob_2
reject_data_temp_3 = reject_data_temp_2[(reject_data_temp_2['prob_2']<0.9)&(reject_data_temp_2['prob_2']>0.2)] 
predict_reject_prob_3,reject_train_3 = reject_data_predict(reject_data_temp_3,pipeline_model_reject_2,feature_used)

reject_evalutaion_detail_3,pipeline_model_userful_3,reject_train_userful_3 = reject_main_train(reject_train_3,\
                                                train_data,predict_reject_prob_3,model_list_eval,feature_used,model_auc_reject_1,0.8,0.2,diff=0)

eval_reject_df_3 = pd.DataFrame(data = reject_evalutaion_detail_3,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])
temp_3 = eval_reject_df_3.merge(eval_df,on=[eval_reject_df_3.index],how='inner')
result_save(temp_3,name = u'0.8_0.2_深度4_0913-1115_3轮0.7437')

# 第三轮保存这一轮的结果
pipeline_model_reject_3 = pipeline_model_userful_3[0]
userful_reject_data_3 = reject_train_userful_3[0]
from sklearn.externals import joblib
joblib.dump(pipeline_model_reject_3,r'D:\Reject_Inference_project\model_pkl_reject_data\best_model_3_0.7437.pkl',compress=3)
userful_reject_data_3.to_csv(r'D:\Reject_Inference_project\model_pkl_reject_data\best_data_3_0.7437.csv',index=False)

