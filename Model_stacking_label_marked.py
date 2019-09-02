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
    lr_model = LogisticRegression().set_params(**param_dict.get('lr',{}))
    if types == 'train':
        RF_model = RandomForestClassifier().set_params(**param_dict.get('RF',{}))
        NB_model = GaussianNB().set_params(**param_dict.get('NB',{}))
        catboost_model = CatBoostClassifier().set_params(**param_dict.get('catboost',{}))
        lightgbm_model = gbm.LGBMClassifier().set_params(**param_dict.get('lightgbm',{}))
        xgboost_model = XGBClassifier().set_params(**param_dict.get('xgboost',{}))
        model_list = [lightgbm_model,xgboost_model,RF_model,NB_model,lr_model,catboost_model]
    elif types == 'eval':
        model_list = [lr_model]
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
        model_fuse = model_list[0]
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
def reject_data_masked(reject_train,predict_reject_prob,n_sample=1000,ratio=1):
    reject_train['user_prob'] = predict_reject_prob
    
    # 好样本与坏样本
    reject_train_temp = reject_train[((reject_train['user_prob']<1)&(reject_train['user_prob']>0.8))\
                                     |((reject_train['user_prob']<0.25)&(reject_train['user_prob']>0.1))]
    
    reject_train_temp['user_type'] = reject_train_temp['user_prob'].apply(lambda x: 1.0 if x>0.5 else 0)
    print('拒绝样本:',reject_train_temp['user_type'].value_counts())
    
    # 训练样本
    temp_1 = reject_train_temp[reject_train_temp['user_type'] == 1].sample(n_sample)
    temp_0 = reject_train_temp[reject_train_temp['user_type'] == 0].sample(ratio*n_sample)
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
                      feature_used,model_auc_raw,model_auc_test,diff=0):  
    reject_evalutaion_detail = {}
    pipeline_model_userful = []
    reject_train_userful = []
    for ratio in [1,2,3]:
        for n_sample in [1000,2000,3000]:    
            print('开始 %s_%s'%(ratio,n_sample))
            # 选取拒绝标签
            reject_train_combine_label = reject_data_masked(reject_train,\
                                                            predict_reject_prob,n_sample,ratio)
            # 合并数据训练集
            X_train_reject_1,y_train_reject_1 = raw_combine_reject(train_data,\
                                                                   reject_train_combine_label,feature_used)
            # 拒绝样本训练
            pipeline_model_reject_1 = fuse_model_train(X_train_reject_1,y_train_reject_1,\
                                        model_list,category_feature,numeric_feature,classifier_type='eval')
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
 

# 保存
def result_save(result):
    writer = pd.ExcelWriter(r'D:\Reject_Inference_project\output\test_process_conclusion_1.xlsx',engine='xlsxwriter')
    result.to_excel(excel_writer=writer,sheet_name=u'阈值_0.75-0.95_0.1-0.25_深度2',index=False)
    writer.save()

# 1 读取特征
train_data,valid_data,test_data,reject_data,feature_dict = load_data()
reject_data_temp = reject_data[(reject_data['risk_time']>'2018-09-13 00:00:00')\
                               &(reject_data['risk_time']<'2018-11-15 00:00:00')]
category_feature = feature_dict['category_var']
numeric_feature = feature_dict['numeric_var']
X_train = train_data[category_feature+numeric_feature]
y_train = train_data['user_type']
X_valid = valid_data[category_feature+numeric_feature]
y_valid = valid_data['user_type']
print('X_train:',X_train.shape,'X_valid:',X_valid.shape)

# 2 模型生成
param_dict_train = {
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
         'colsample_bytree':0.8
        },
    'lr':{
        'C':0.5,
        'penalty':'l1',
        'random_state':10,
        'class_weight':{0:1,1:3}
        },
    'RF':{
        'n_estimators':700,
        'max_depth':3,
        'random_state':10,
        'class_weight':{0:1,1:3}
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

    
param_dict_eval =  {
    'lr':{
        'C':0.1,
        'penalty':'l1',
        'random_state':8,
        }
    }
model_list_base = model_generate(param_dict_train,types='train')
model_list_eval = model_generate(param_dict_eval,types='eval')

# 原始模型结果
pipeline_model_base = fuse_model_train(X_train,y_train,model_list_base,category_feature,\
                     numeric_feature,classifier_type='train')
pipeline_model_eval = fuse_model_train(X_train,y_train,model_list_eval,category_feature,\
                     numeric_feature,classifier_type='eval')

feature_used = category_feature + numeric_feature
## 原子模型输出
#submodel_raw_result = submodel_evaluation(train_data,valid_data,model_list,\
#                        category_feature,numeric_feature)

# 原始模型评估
evaluation_raw = {}
evaluation_raw_details = model_fuse_evaluation(pipeline_model_eval,train_data,valid_data,test_data,feature_used)
evaluation_raw['raw'] = evaluation_raw_details
model_auc_raw = evaluation_raw_details[1]
model_auc_test = evaluation_raw_details[3]
eval_df = pd.DataFrame(data = evaluation_raw,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])

# 预测拒绝标签
predict_reject_prob,reject_train = reject_data_predict(reject_data,pipeline_model_base,feature_used)

# 原始模型生成第一批拒绝样本测试：单轮测试
reject_evalutaion_detail,pipeline_model_userful,reject_train_userful = reject_main_train(reject_train,\
                                                train_data,predict_reject_prob,model_list_eval,feature_used,model_auc_raw,model_auc_test,diff=0)

# 输出最后结果
eval_reject_df_1 = pd.DataFrame(data = reject_evalutaion_detail,index=['valid_ks','valid_auc',\
                                          'ks_test','auc_test','auc_train','ks_train','accuracy'])
temp = eval_reject_df_1.merge(eval_df,on=[eval_reject_df_1.index],how='inner')
result_save(temp)    
 
## 第一批模型生成第二批拒绝样本测试：二轮测试
#reject_evalutaion_detail,pipeline_model_userful,reject_train_userful = reject_main_train(reject_train,\
#                                                train_data,predict_reject_prob,feature_used,model_auc_raw,model_auc_test,diff=0)






# 子模型结果输出
#model_result_dict = submodel_evaluation(train_data,valid_data,model_list,feature_dict)


#from pgy_model import pipe_train_test_evaluate
#from pgy_model import model_result_combine
#
#data_dict = {'train':{'X':train_data[numeric_feature+category_feature],'y':train_data['user_type']},
#             'valid_public':{'X':valid_data[numeric_feature+category_feature],'y':valid_data['user_type']},
#             'test_private':{'X':test_data[numeric_feature+category_feature],'y':test_data['user_type']}
#            }
#model_detail_result = {}
#model_statistic_result = {}
#
#model_detail_result,model_statistic_result = pipe_train_test_evaluate(data_dict,pipeline_model)
#model_predict_result = model_result_combine({'model_fuse':model_detail_result},'valid_public')
#
## ks曲线
#ks = plot_ks_curve(model_predict_result.get('model_fuse').get('predict'),model_predict_result.get('model_fuse').get('true'),n=10,return_graph=True)
## roc曲线
#roc_dict,auc_dict = plot_multi_roc_curve_dict_type(model_predict_result)
## 通过率vs拒绝率曲线
#bad_rate_result = plot_multi_reject_bad_curve_dict_type(model_predict_result)
## PR曲线
#plot_multi_PR_curve_dict_type(model_predict_result)
## 预测概率目的曲线
#plot_density_curve(model_predict_result.get('model_fuse').get('true'),model_predict_result.get('model_fuse').get('predict'))


