# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold,train_test_split
from pgy_utils import timecount
from pgy_evaluation import plot_ks_curve,plot_roc_curve
from bayes_opt import BayesianOptimization
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn2pmml.decoration import ContinuousDomain,CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import gen_features
from sklearn.pipeline import FeatureUnion,Pipeline
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.utils import indexable
from pgy_bins import *


def gbdt_feature_importance(data_dict, \
                            gbdt_estimator, \
                            importance_type='gain', \
                            category_feature=[]):
    '''
    计算一组数据集在gbdt_estimator下面的特征重要度
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    gbdt_estimator: gbdt模型
    category_feature: 需要处理的类别型变量
    
    return:
    feature_importance_detail: 特征重要度 map对象 每个数据集key下面对应的一个pd.Series对象
    '''
    ##存储不同数据集下的特征重要度
    feature_importance_detail = {}
    ##进行模型训练
    gbdt_estimator.set_params(**{'importance_type':importance_type})
    for key in data_dict.keys():
        print('正在进行数据集{0}的gbdt特征重要度计算工作'.format(key))
        temp_X = data_dict.get(key).get('X')
        temp_y = data_dict.get(key).get('y')

        ##是否要处理类别型特征
        if len(category_feature) == 0:
            gbdt_estimator.fit(temp_X,temp_y)
        else:
            gbdt_estimator.fit(temp_X,temp_y,categorical_feature = category_feature,verbose=False) 
        feature_importance_detail[key] = pd.Series(gbdt_estimator.feature_importances_,index=temp_X.columns)
    ##取出特征重要度
    return feature_importance_detail


def gbdt_mix_feature_importance(data_dict, \
                                gbdt_estimator_map, \
                                importance_type_list=['gain'], \
                                category_feature=[], \
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10), \
                                groups=None, \
                                use_validation=True):
    '''
    计算一组数据集在同一类型多组参数的gbdt_estimator、不同importance_type以及数据cv划分下的特征重要度
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    gbdt_estimator_map: 多个gbdt模型组成的map
    importance_type_list: 多种importance_type组成的list\
    category_feature: 需要处理的类别型变量
    cv: 数据集切分方法 用于train的切分
    groups: 数据分组 用于trian的分组
    use_validation: 使用验证集还是训练集进行特征重要度的计算
    
    return:
    feature_importance_detail: 特征重要度 map对象 每个importance_type的key、gbdt_estimator的key、数据集key下面对应的一个pd.Series对象
    '''
    print('开始计算特征重要度'.center(50, '='))
    ##为了防止下面的操作更改dict内容所以复制一份 但是考虑到内存占用问题后续可以优化
    data_dict = data_dict.copy()
    ##取出训练集
    X = data_dict.get('train').get('X')
    y = data_dict.get('train').get('y')
    ##存储结果
    feature_importance_detail = {}
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):
        if groups is None:
            group_fold = 'NULL'
        else:
            group_fold = str(groups[valid_index[0]])
        print('正在进行第{}折的验证,验证组号为{}'.format(fold_n,group_fold))
        ##取出当前轮使用的的训练数据和验证数据
        X_valid,y_valid = X.iloc[valid_index],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        ##用于训练的数据集准备
        if use_validation:
            temp_dict = {'fold_'+str(fold_n)+'_group_'+group_fold:{'X':X_valid,'y':y_valid}}
        else:
            temp_dict = {'fold_'+str(fold_n)+'_group_'+group_fold:{'X':X_train,'y':y_train}}
        data_dict.update(temp_dict)
    ##遍历gbdt_estimator_list和importance_type_list
    for importance_type in importance_type_list:
        temp_feature_importance_detail = {}
        for gbdt_estimator_key in gbdt_estimator_map.keys():
            print('正在使用进行{0}进行类型为{1}的gbdt特征重要度计算工作'.format(gbdt_estimator_key,importance_type))
            temp_feature_importance_detail[gbdt_estimator_key] = gbdt_feature_importance(data_dict,\
                                                                                         gbdt_estimator_map.get(gbdt_estimator_key),\
                                                                                         importance_type,
                                                                                         category_feature)
        feature_importance_detail[importance_type] = temp_feature_importance_detail
    return feature_importance_detail


@timecount()
def IV_feature_importance(data_dict, \
                          category_feature=[],\
                          feature_bins_dict={}, \
                          special_attribute_dict={}, \
                          binning_param={'numeric':{'max_interval':10,'method':'ChiMerge','woe_shift':0,"tree_params":{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0}},
                                         'category':{'max_interval':10,'method':'ChiMerge','woe_shift':0,"tree_params":{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0}}},\
                          calculate_cp=True
                          ):
    '''
    计算一组数据集的特征IV
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    category_feature: 要进行类别型分箱处理的特征
    feature_bins_dict: 特征分箱的分箱方案 每个特征名对应其自己的分箱细节 map
    special_attribute_dict: 每个特征名对应的特殊值（一般是缺失值添补值） map
    binning_param: 分箱的具体参数 比如连续型变量和类别型变量分别使用什么样的分箱方法 max_interval是多少 决策树参数是多少
    calculate_cp: 是否计算分箱节点操作 如果给出的话就会对训练集上的数据进行分箱节点的计算工作
    
    
    return:
    feature_importance_detail: 特征IV map对象 每个数据集key下面对应的一个pd.Series对象
    feature_importance_bins_detail: 特征的分箱细节 map对象 每个数据集key以及每个变量key下面对应一个pd.DataFrame对象
    feature_bins_dict: 最终的特征分箱方案 每个特征名对应其自己的分箱细节 map
    special_attribute_dict: 最终的每个特征名对应的特殊值（一般是缺失值添补值） map
    '''
    ##存储不同数据集下的特征重要度
    feature_importance_detail = {}
    feature_importance_bins_detail = {}
    ##如果calculate_cp为True，那么就用训练集来训练这个feature_bins_dict（只计算feature_bins_dict中没有给的变量），如果给了calculate_cp为False，那么就直接用feature_bins_dict里面的值进行分箱woe、IV的计算工作
    ##计算IV不能有缺失值 所以缺失值要么自己预先进行了添补 如果没有添补 那么就会进行相应的默认填补的方法 然后返回special_attribute_dict
    ##所以如果不想进行单独的特殊值处理操作 那么就预先进行缺失值添补 然后不要给出special_attribute_dict
    if calculate_cp:
        ##获取训练集的数据
        X = data_dict.get('train').get('X')
        y = data_dict.get('train').get('y')
        for var in X.columns:
            ##如果feature_bins_dict的key包含该变量 那么就不进行相应的分箱计算工作
            if var in feature_bins_dict.keys():
                continue
            ##取出var和label
            df = X[[var]].copy()
            df['user_type'] = y
            ##计算缺失值个数
            na_counts = np.sum(df[var].isna())
            ##计算值个数
            value_counts = len(df[var].value_counts())
            ##如果只有0个值或者1个值 那么直接给出分箱结果
            if value_counts <= 1:
                bins_df = pd.DataFrame()
                IV = 0
                bins = None
                special_attribute = []
                feature_bins_dict[var] = bins
                special_attribute_dict[var] = special_attribute
            ##如果是类别型特征
            elif var in category_feature:
                ##取出分箱参数
                max_interval = binning_param.get('category').get('max_interval',10)
                ##如果value_counts小于max_interval就用default一个类型一个箱 否则则取给定的分箱方法 默认值为ChiMerge
                method = ['default', binning_param.get('category').get('method','ChiMerge')][value_counts>max_interval]
                special_attribute = special_attribute_dict.get(var,[])
                tree_params = binning_param.get('category').get('tree_params',{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0})
                woe_shift =  binning_param.get('category').get('woe_shift',0.000001)
                 ##进行缺失值添补判断
                if len(special_attribute)>1:
                    raise ValueError("detect special_attribute {0} for feature {1} which contains more than 1 special elements".format(special_attribute,var))   
                if na_counts > 0 and special_attribute != []:
                    ##如果有给出填补方案 那么就用填补方案进行添补 但是要提醒用户
                    print("detect nan values in {0} when got special_attribute {1}".format(var, special_attribute))
                    df = df.fillna(special_attribute[0])
                elif na_counts > 0 and special_attribute == []:
                    special_attribute = ['未知']
                    df = df.fillna('未知')
                ##进行分箱操作并存入dict当中
                bins_df, IV, bins = category_var_binning(df,var,'user_type',max_interval=max_interval,method=method,special_attribute=special_attribute,tree_params=tree_params,woe_shift=woe_shift)
                feature_bins_dict[var] = bins
                special_attribute_dict[var] = special_attribute
            ##如果是数值变量
            else:
                ##取出分箱参数
                max_interval = binning_param.get('numeric').get('max_interval',10)
                 ##如果value_counts小于max_interval就用差值分箱一个数值一个箱 否则则取给定的分箱方法 默认值为ChiMerge
                method = ['Interpolate', binning_param.get('numeric').get('method','ChiMerge')][value_counts>max_interval]
                special_attribute = special_attribute_dict.get(var,[])
                tree_params = binning_param.get('numeric').get('tree_params',{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0})
                woe_shift =  binning_param.get('numeric').get('woe_shift',0.000001)
                ##进行缺失值添补判断
                if len(special_attribute)>1:
                    raise ValueError("detect special_attribute {0} for feature {1} which contains more than 1 special elements".format(special_attribute,var))   
                if na_counts > 0 and special_attribute != []:
                    ##如果有给出填补方案 那么就用填补方案进行添补 但是要提醒用户
                    print("detect nan values in {0} when got special_attribute {1}".format(var, special_attribute))
                    df = df.fillna(special_attribute[0])
                elif na_counts > 0 and special_attribute == []:
                    special_attribute = [df[var].min()-1]
                    df = df.fillna(df[var].min()-1)
                ##进行分箱操作并存入dict当中
                bins_df, IV, bins = numeric_var_binning(df,var,'user_type',max_interval=max_interval,method=method,special_attribute=special_attribute,tree_params=tree_params,woe_shift=woe_shift)
                feature_bins_dict[var] = bins
                special_attribute_dict[var] = special_attribute
    ##如果给出了feature_bins_dict那么直接按照bin对不同数据集下面不同变量进行分箱操作即可
    else:
        ##遍历数据集
        for key in data_dict.keys():
            temp_X = data_dict.get(key).get('X')
            temp_y = data_dict.get(key).get('y')
            temp_feature_importance_detail = {}
            temp_feature_importance_bins_detail = {}
            ##遍历变量
            for var in feature_bins_dict.keys():
                ##取出var和label
                df = temp_X[[var]].copy()
                df['user_type'] = temp_y
                ##取出分箱方案 进行复制操作 后续需要可能要进行修改
                bins = feature_bins_dict.get(var,None)
                ##如果没有给出分箱 那么就是只有0个值或者1个值的变量 直接给出IV和分箱细节
                if bins is None:
                    bins_df = pd.DataFrame()
                    IV = 0
                    temp_feature_importance_detail[var]=IV
                    temp_feature_importance_bins_detail[var]=bins_df
                    continue
                ##取出特殊值（用于缺失值添补，因为经过第一步处理时候如果有缺失值，那么就会补上缺失值然后赋值进special_attribute，所里这里用special_attribute进行缺失值）
                special_attribute = special_attribute_dict.get(var,[])
                if len(special_attribute)>1:
                    raise ValueError("detect special_attribute {0} for feature {1} which contains more than 1 special elements".format(special_attribute,var))   
                if special_attribute != []:
                    ##进行添补
                    df = df.fillna(special_attribute[0])
                ##检测是否还有缺失值
                ##计算缺失值个数
                na_counts = np.sum(df[var].isna())
                if na_counts > 0:
                    raise ValueError("detect nan values in {0}".format(var))   
                ##分箱类型是数值分箱
                if isinstance(bins,list):
                    ##获取woe参数的
                    woe_shift =  binning_param.get('numeric').get('woe_shift',0.000001)
                    bins_df, IV, bins = numeric_var_binning_with_bins(df, var, 'user_type', bins=bins, woe_shift=woe_shift)
                    temp_feature_importance_detail[var]=IV
                    temp_feature_importance_bins_detail[var]=bins_df
                ##分箱类型是类别分箱
                elif isinstance(bins,dict):
                    ##如果是进行类别型变量的处理 把df对应列的类型改成object 不然会报错
                    df[var] = df[var].astype(object)
                    woe_shift =  binning_param.get('category').get('woe_shift',0.000001)
                    bins_df, IV, bins = category_var_binning_with_bins(df, var, 'user_type', bins=bins, woe_shift=woe_shift)
                    temp_feature_importance_detail[var]=IV
                    temp_feature_importance_bins_detail[var]=bins_df
                else:
                    raise ValueError("unexpected bins {0}".format(bins))
            feature_importance_detail[key]=pd.Series(temp_feature_importance_detail)
            feature_importance_bins_detail[key]=temp_feature_importance_bins_detail
    ##如果feature_bins_dict没有给 那么就是计算feature_bins_dict和special_attribute的过程 返回空的特征重要度和特征重要度分箱明细woe信息 以及处理好的分箱方案信息以及缺失值添补信息
    ##如果给出了feature_bins_dict信息 那么就是按照feature_bins_dict和special_attribute进行分箱IV计算以及分箱细节的计算
    return feature_importance_detail,feature_importance_bins_detail,feature_bins_dict,special_attribute_dict



@timecount()
def IV_mix_feature_importance(data_dict, \
                              category_feature=[],\
                              feature_bins_dict={}, \
                              special_attribute_dict={}, \
                              binning_param={'numeric':{'max_interval':10,'method':'ChiMerge','woe_shift':0,"tree_params":{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0}},
                                             'category':{'max_interval':10,'method':'ChiMerge','woe_shift':0,"tree_params":{"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.001,"random_state":0}}},\
                              cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),\
                              groups=None,\
                              use_validation=True):
    '''
    计算一组数据集在cv划分下的IV 通过训练集进行分箱方法的选取 然后在不同的train上的cv数据集上以及
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    category_feature: 要进行类别型分箱处理的特征
    feature_bins_dict: 特征分箱的分箱方案 每个特征名对应其自己的分箱细节 map
    special_attribute_dict: 每个特征名对应的特殊值（一般是缺失值添补值） map
    binning_param: 分箱的具体参数 比如连续型变量和类别型变量分别使用什么样的分箱方法 max_interval是多少 决策树参数是多少
    cv: 数据集切分方法 用于train的切分
    groups: 数据分组 用于trian的分组
    use_validation: 使用验证集还是训练集进行特征重要度的计算
    
    return:
    feature_importance_detail: 特征IV map对象 每个数据集key下面对应的一个pd.Series对象
    feature_importance_bins_detail: 特征的分箱细节 map对象 每个数据集key以及每个变量key下面对应一个pd.DataFrame对象
    feature_bins_dict: 最终的特征分箱方案 每个特征名对应其自己的分箱细节 map
    special_attribute_dict: 最终的每个特征名对应的特殊值（一般是缺失值添补值） map
    '''
    print('开始计算特征IV'.center(50, '='))
    ##为了防止下面的操作更改dict内容所以复制一份 但是考虑到内存占用问题后续可以优化
    data_dict = data_dict.copy()
    ##进行分箱节点的计算
    feature_importance_detail,feature_importance_bins_detail,feature_bins_dict,special_attribute_dict = IV_feature_importance(data_dict, \
                                                                                                                         category_feature=category_feature,\
                                                                                                                         feature_bins_dict=feature_bins_dict, \
                                                                                                                         special_attribute_dict=special_attribute_dict, \
                                                                                                                         binning_param=binning_param, \
                                                                                                                         calculate_cp=True
                                                                                                                         )
    ##取出训练集
    X = data_dict.get('train').get('X')
    y = data_dict.get('train').get('y')
    ##存储结果
    feature_importance_detail = {}
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):
        if groups is None:
            group_fold = 'NULL'
        else:
            group_fold = str(groups[valid_index[0]])
        print('正在进行第{}折的验证,验证组号为{}'.format(fold_n,group_fold))
        ##取出当前轮使用的的训练数据和验证数据
        X_valid,y_valid = X.iloc[valid_index],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        ##用于训练的数据集准备
        if use_validation:
            temp_dict = {'fold_'+str(fold_n)+'_group_'+group_fold:{'X':X_valid,'y':y_valid}}
        else:
            temp_dict = {'fold_'+str(fold_n)+'_group_'+group_fold:{'X':X_train,'y':y_train}}
        data_dict.update(temp_dict)
    ##通过给定的分箱节点和缺失值添补方案 进行IV和分箱细节计算
    feature_importance_detail,feature_importance_bins_detail,feature_bins_dict,special_attribute_dict = IV_feature_importance(data_dict, \
                                                                                                                         category_feature=category_feature,\
                                                                                                                         feature_bins_dict=feature_bins_dict, \
                                                                                                                         special_attribute_dict=special_attribute_dict, \
                                                                                                                         binning_param=binning_param,\
                                                                                                                         calculate_cp=False
                                                                                                                         )
    return feature_importance_detail,feature_importance_bins_detail,feature_bins_dict,special_attribute_dict



def gbdt_cv_evaluate_earlystop(data_dict, gbdt_estimator, total_features, category_features, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10), groups=None):
    '''
    使用gbdt模型进行交叉验证评估，训练中使用earlystop，适用于XBG、GBM、GBDT以及CAT等模型
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    gbdt_estimator: gbdt类的estimator
    total_features: 入模所有特征 list
    category_features: 入模类别特征 list
    cv: 数据集切分方法
    groups: 数据分组
    
    return:
    fold_detail_result: 每一折内各个数据集预测的概率结果和真实标签
    fold_statistic_result: 每一折内各个数据集预测的指标
    fold_best_iteration_result: 每一折内的提前停止情况
    '''
    ##为了防止下面的操作更改dict内容所以复制一份 但是考虑到内存占用问题后续可以优化
    data_dict = data_dict.copy()
    ##取出训练集 进行模型训练
    X = data_dict.get('train').get('X')
    y = data_dict.get('train').get('y')
    fold_detail_result = {}
    fold_statistic_result = {}    
    fold_best_iteration_result = {}
    
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):
        if groups is None:
            group_fold = 'NULL'
        else:
            group_fold = str(groups[valid_index[0]])
        print('正在进行第{}折的验证,验证组号为{}'.format(fold_n,group_fold))
        ##取出当前轮使用的的训练数据和验证数据
        X_outer_valid,y_outer_valid = X.iloc[valid_index],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        ##对train进行数据集划分用于early stop
        X_train, X_inner_valid, y_train, y_inner_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
        ##用于训练的数据集准备
        temp_dict = {'train':{'X':X_train,'y':y_train},
                     'inner_valid':{'X':X_inner_valid,'y':y_inner_valid},
                     'outer_valid':{'X':X_outer_valid,'y':y_outer_valid}}
        data_dict.update(temp_dict)
        ##模型训练
        #gbdt_estimator.set_params({'n_estimators':20000})
        if category_features == []:
            gbdt_estimator.fit(X_train[total_features], y_train, eval_metric=['auc'], eval_set = [(X_inner_valid[total_features],y_inner_valid)],early_stopping_rounds=500,verbose=500)
        else:
            gbdt_estimator.fit(X_train[total_features], y_train, categorical_feature=category_features, eval_metric=['auc'], eval_set = [(X_inner_valid[total_features],y_inner_valid)],early_stopping_rounds=500,verbose=500)            
        ##获取最佳训练论数
        if hasattr(gbdt_estimator,'best_iteration_'):
            best_iteration = gbdt_estimator.best_iteration_
        elif hasattr(gbdt_estimator,'best_iteration'):
            best_iteration = gbdt_estimator.best_iteration
        else:
            raise ValueError("cannot find best_iteration in {0}".format(gbdt_estimator))   
        ##进行模型预测
        detail_result = {}
        statistic_result = {}
        for key in data_dict.keys():
            temp_X = data_dict.get(key).get('X')
            temp_y = data_dict.get(key).get('y')
            temp_predict = gbdt_estimator.predict_proba(temp_X[total_features])[:,1]
            temp_auc = plot_roc_curve(temp_y,temp_predict)
            temp_ks = plot_ks_curve(temp_predict, temp_y, is_score=False, n=10)
            detail_result[key] = {'predict':temp_predict,'true':temp_y.values}
            statistic_result[key] = {'auc':temp_auc,'ks':temp_ks}
        ##数据存储
        fold_detail_result['fold_'+str(fold_n)+'_group_'+group_fold] = detail_result
        fold_statistic_result['fold_'+str(fold_n)+'_group_'+group_fold] = statistic_result
        fold_best_iteration_result['fold_'+str(fold_n)+'_group_'+group_fold] = best_iteration
    return fold_detail_result,fold_statistic_result,fold_best_iteration_result




def gbdt_feature_selector(data_dict,\
                          gbdt_estimator,\
                          feature_rank, \
                          category_features, \
                          cv_list=[StratifiedKFold(n_splits=5, shuffle=True, random_state=0)],\
                          groups_list=[None],\
                          weights_list=[1],\
                          rounds=100,\
                          step=1,\
                          auc_diff_threshold=0,\
                          auc_initial=0.55):
    '''
    逐步加入变量的stepwise特征筛选函数 使用lightgbm模型
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    gbdt_estimator: gbdt类的estimator
    feature_rank: 搜索特征的按照各种方式排序后的list
    category_features: 需要当做类别型变量处理的特征
    cv_list: 数据集切分方法组成的list 默认为5折StratifiedKFold
    groups_list: 数据集切分方法的参数组成的list 默认不给
    weight_list: 不同数据切分方法最终评估结果的权重
    rounds: 总共筛选多少轮
    step: 每轮加入多少个变量
    auc_diff_threshold: auc有多少提升才会被选入到其中
    auc_initial: 第一个加入的变量至少需要达到多高的auc才能进入模型
    
    return:
    feature_selected 最终入模特征
    step_detail 每一轮cv的细节内容构成的map key为roundx x为论数
    step_outer_valid_statistic 被选入论的outer_valid评估结果
    '''
    ##进行一些预置检查
    ##检查3个list的长度是否相等
    cv_list,groups_list,weights_list = indexable(cv_list,groups_list,weights_list)
    ##进行一些准备工作
    print("开始特征筛选".center(50, '='))
    each_round_start = range(0,rounds*step,step)
    ##每一轮评估之前已经入模的特征
    feature_selected = []
    ##每一轮的明细数据
    step_detail = {}
    ##存储最终入模的特征数据
    step_outer_valid_statistic = {}
    for i in each_round_start:
        print('*************rounds: %d****************'%(i/step+1))
        ##选出特征
        feature_added = feature_rank[i:i+step]
        feature_used = feature_selected + feature_added
        category_feature_uesd = [i for i in feature_used if i in category_features]
        print('入模型特征数:'+str(len(feature_used)))
        print('当前轮数考察特征:',feature_added)  
        ##存储每一个CV下的结果
        cv_detail_result = {}
        cv_statistic_result = {}
        cv_best_iteration_result = {}
        cv_statistic = {}
        cv_outer_valid_statistic = {}
        for cv_index in range(len(cv_list)):
            fold_detail_result,fold_statistic_result,fold_best_iteration_result = gbdt_cv_evaluate_earlystop(data_dict=data_dict, gbdt_estimator=gbdt_estimator, total_features=feature_used, category_features=category_feature_uesd, cv=cv_list[cv_index], groups=groups_list[cv_index])
            cv_detail_result[cv_index] = fold_detail_result
            cv_statistic_result[cv_index] = fold_statistic_result
            cv_best_iteration_result[cv_index] = fold_best_iteration_result
            ##取出其中的评估数
            fold_statistic = pd.DataFrame()
            for fold_key in fold_statistic_result.keys():
                temp_statistic = pd.DataFrame(fold_statistic_result[fold_key]).T
                temp_statistic.columns = [fold_key+'_']+temp_statistic.columns
                fold_statistic = pd.concat([fold_statistic,temp_statistic],axis=1)
            ##包含各种评估指标的列
            evaluation_map = {}
            ##包含auc数据的列
            evaluation_map['auc'] = [i for i in fold_statistic.columns if '_auc' in i]
            ##包含ks数据的列
            evaluation_map['ks'] = [i for i in fold_statistic.columns if '_ks' in i]
            ##计算ks数据的均值和方差
            for evaluation_key in evaluation_map.keys():
                fold_statistic[evaluation_key+'_mean'] = fold_statistic[evaluation_map[evaluation_key]].apply(lambda x:np.mean(x),axis=1)
                fold_statistic[evaluation_key+'_std'] = fold_statistic[evaluation_map[evaluation_key]].apply(lambda x:np.std(x),axis=1)
            cv_statistic[cv_index] = fold_statistic
            cv_outer_valid_statistic[cv_index] = fold_statistic.loc['outer_valid',[i+'_mean' for i in evaluation_map.keys()]+[i+'_std' for i in evaluation_map.keys()]]
        ##进行cv结果的评估 使用的是outer_valid的数据 对不同cv下面的结果进行加权
        outer_valid_statistic = pd.DataFrame(cv_outer_valid_statistic).apply(lambda x:np.dot(x,np.array(weights_list)),axis =1)
        ##存储每一轮结果的明细数据
        current_step_detail = {}              
        current_step_detail['auc_threshold'] = auc_initial
        current_step_detail['feature_initial'] = feature_selected
        current_step_detail['feature_added'] = feature_added
        current_step_detail['feature_used'] = feature_used
        current_step_detail['category_feature_used'] = category_feature_uesd
        current_step_detail['cv_detail_result'] = cv_detail_result
        current_step_detail['cv_best_iteration_result'] = cv_best_iteration_result
        current_step_detail['cv_statistic'] = cv_statistic
        current_step_detail['cv_outer_valid_statistic'] = cv_outer_valid_statistic
        current_step_detail['outer_valid_statistic'] = outer_valid_statistic
        current_step_detail['is_delete'] = (outer_valid_statistic['auc_mean']-auc_initial) < auc_diff_threshold
        step_detail['round'+str(int(i/step+1))] = current_step_detail
        print('当前step下加权评估结果为{0},auc阈值为{1}(包含每部最低提升要求{2})'.format(outer_valid_statistic,auc_initial+auc_diff_threshold,auc_diff_threshold))
        ##不满足条件 就不把当前变量加入到候选集当中
        if (outer_valid_statistic['auc_mean']-auc_initial) < auc_diff_threshold:
            print('删除当前批次的入模特征')
            continue
        ##满足条件 加入到候选集当中
        feature_selected = feature_used
        ##更新阈值
        auc_initial = outer_valid_statistic['auc_mean']
        ##存储每一步加入到模型单重
        step_outer_valid_statistic['round'+str(int(i/step+1))] = outer_valid_statistic
    return feature_selected,step_detail,step_outer_valid_statistic


def bayes_parameter_opt_lgb(X, y, category_feature, init_round=5, opt_round=10, n_folds=5, random_seed=0, n_estimators=10000,
                            learning_rate=0.05):
    '''
    贝叶斯超参数筛选 lightgbm版
    X: 训练X
    y: 训练y
    category_feature: 类别特征（）
    init_round: 最开始随机搜索的次数
    opt_round: 贝叶斯优化搜搜的次数
    n_folds: CV的折数
    random_seed: 随机种子
    n_estimators: 模型树个数上限
    learning_rate: 学习率
    
    return:
    最优的参数
    
    
    调用例子：
    lgb_opt_params = bayes_parameter_opt_lgb(X, y,category_feature,init_round=5, opt_round=15, n_folds=5, random_seed=0, n_estimators=10000,
                                     learning_rate=0.05)
    '''
    # prepare data
    train_data = lgb.Dataset(data=X, label=y,categorical_feature=category_feature, free_raw_data=False)

    # parameters
    def lgb_eval(num_leaves, max_depth, reg_alpha, reg_lambda, min_split_gain,min_child_weight,max_bin):
        params = {'application': 'binary', 'num_iterations': n_estimators, 'learning_rate': learning_rate, 
                  'early_stopping_round': 100, 'metric': 'auc'}
        params["nthread"] = 4
        params["num_leaves"] = int(round(num_leaves))
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params["silent"] = -1
        params["verbose"] = -1
        params['max_bin'] = int(round(max_bin))
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, verbose_eval=200,
                           metrics=['auc'])
        return max(cv_result['auc-mean'])

    # range
    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (12, 50),
                                            'max_bin': (50, 300),
                                            'max_depth': (2, 5),
                                            'reg_alpha': (0, 4),
                                            'reg_lambda': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 70)}, random_state=0)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return lgbBO.max




def bayes_parameter_opt_xgb(X, y, init_round=5, opt_round=10, n_folds=5, random_seed=0, n_estimators=10000,
                            learning_rate=0.05):
    '''
    贝叶斯超参数筛选 xgboost版
    X: 训练X
    y: 训练y
    init_round: 最开始随机搜索的次数
    opt_round: 贝叶斯优化搜搜的次数
    n_folds: CV的折数
    random_seed: 随机种子
    n_estimators: 模型树个数上限
    learning_rate: 学习率
    
    return:
    最优的参数
    
        
    调用例子:
    xgb_opt_params = bayes_parameter_opt_xgb(X, y, init_round=5, opt_round=15, n_folds=5, random_seed=0, n_estimators=10000,
                                     learning_rate=0.1)
    '''
    # prepare data
    train_data =xgb.DMatrix(data=X, label=y)

    # parameters
    def xgb_eval(max_depth, reg_alpha, reg_lambda, gamma,min_child_weight,subsample,colsample_bytree):
        params = {'application': 'binary:logistic','learning_rate': learning_rate,'metric': 'auc'}
        params["nthread"] = 4
        params['max_depth'] = int(round(max_depth))
        params['reg_alpha'] = max(reg_alpha, 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['gamma'] = gamma
        params['min_child_weight'] = min_child_weight
        params['eval_metric'] = 'auc'
        params["silent"] = 1
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        cv_result = xgb.cv(params, train_data,num_boost_round=n_estimators, nfold=n_folds, seed=random_seed, verbose_eval=100,early_stopping_rounds=100,
                       metrics=['auc'],maximize=True)
        print(len(cv_result))
        return max(cv_result['test-auc-mean'])

    # range
    xgbBO = BayesianOptimization(xgb_eval, {'gamma': (0, 5),
                                            'max_depth': (2, 5),
                                            'reg_alpha': (0, 5),
                                            'reg_lambda': (0, 3),
                                            'min_child_weight': (5, 50),
                                            'subsample': (0.5, 1),
                                            'colsample_bytree': (0.1, 1)}, random_state=random_seed)
    # optimize
    xgbBO.maximize(init_points=init_round, n_iter=opt_round)
    # return best parameters
    return xgbBO.max


def pipe_cv_evaluate(data_dict,pipeline_estimator,cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=10),groups=None):
    '''
    单个Pipeline的CV评估的结果 对训练数据进行CV 测试数据作为辅助评估可以不给
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    pipeline_estimator:pipeline模型
    cv: 数据集切分方法
    groups: 数据分组
    
    return:
    fold_detail_result: 每一折内各个数据集预测的概率结果和真实标签
    fold_statistic_result: 每一折内各个数据集预测的指标
    '''
    ##为了防止下面的操作更改dict内容所以复制一份 但是考虑到内存占用问题后续可以优化
    data_dict = data_dict.copy()
    ##取出训练集 进行模型训练
    X = data_dict.get('train').get('X')
    y = data_dict.get('train').get('y')
    fold_detail_result = {}
    fold_statistic_result = {}
    ##遍历数据集
    for fold_n, (train_index, valid_index) in enumerate(cv.split(X,y,groups)):
        if groups is None:
            group_fold = 'NULL'
        else:
            group_fold = str(groups[valid_index[0]])
        print('正在进行第{}折的验证,验证组号为{}'.format(fold_n,group_fold))
        ##取出当前轮使用的的训练数据和验证数据
        X_valid,y_valid = X.iloc[valid_index],y.iloc[valid_index]
        X_train,y_train = X.iloc[train_index],y.iloc[train_index]
        ##用于训练的数据集准备
        temp_dict = {'train':{'X':X_train,'y':y_train},
                     'valid':{'X':X_valid,'y':y_valid}}
        data_dict.update(temp_dict)
        ##进行模型的训练和预测
        detail_result,statistic_result = pipe_train_test_evaluate(data_dict,pipeline_estimator)
        ##数据存储
        fold_detail_result['fold_'+str(fold_n)+'_group_'+group_fold] = detail_result
        fold_statistic_result['fold_'+str(fold_n)+'_group_'+group_fold] = statistic_result
    
    return fold_detail_result,fold_statistic_result

def pipe_train_test_evaluate(data_dict,pipeline_estimator):
    '''
    进行pipeline在训练集和多个测试集上面的评估
    data_dict: 多个数据集组成的dict 包含train test_xxx等等key
    pipeline_estimator: pipeline模型
    
    return:
    detail_result: 预测的概率结果和真实标签
    statistic_result: 预测的指标
    '''
    ##取出训练集 进行模型训练
    X_train = data_dict.get('train').get('X')
    y_train = data_dict.get('train').get('y')
    pipeline_estimator.fit(X_train,y_train)
    statistic_result = {}
    detail_result = {}
    ##进行模型预测
    for key in data_dict.keys():
        temp_X = data_dict.get(key).get('X')
        temp_y = data_dict.get(key).get('y')
        temp_predict = pipeline_estimator.predict_proba(temp_X)[:,1]
        temp_auc = plot_roc_curve(temp_y,temp_predict)
        temp_ks = plot_ks_curve(temp_predict, temp_y, is_score=False, n=10)
        detail_result[key] = {'predict':temp_predict,'true':temp_y.values}
        statistic_result[key] = {'auc':temp_auc,'ks':temp_ks}
    return detail_result,statistic_result


## 构建pipeline模型
def make_pipeline_model(numeric_feature,category_feature,estimator,X=None,y=None):
    '''
    通过指定类别型和数值型特征构建以及指定的模型构建pipeline,如果给出数据集就完成训练,最终返回pipeline模型
    numeric_feature: 数值特征 list
    category_feature: 类别特征 list
    X:X数据 传入pandas.DataFrame对象
    y:Y数据 传入pandas.Series对象
    
    return:
    pipeline_model
    '''
    feature_def = gen_features(
        columns=category_feature,
        classes=[CategoricalDomain,CategoricalImputer,LabelBinarizer]
        )
    mapper_numerical = DataFrameMapper([
            (numeric_feature,[ContinuousDomain(),SimpleImputer(strategy='mean'),StandardScaler()])
    ])
    mapper_category = DataFrameMapper(feature_def)
    ##判断是否有类别特征
    if category_feature == []:
        mapper = mapper_numerical
    else:
        mapper = FeatureUnion([('mapper_numerical',mapper_numerical),('mapper_category',mapper_category)])
    ##构建pipeline
    pipeline_model = PMMLPipeline([
        ('mapper',mapper),
        ('classifier',estimator)
    ])
    if X is not None and y is not None:
        pipeline_model.fit(X,y)
    return pipeline_model




def model_result_combine(model_result_dict,data_name):
    """
    把多个模型在多个数据集上预测的结果组成的dict中特定的数据集拿出来
    model_result_dict多个模型在多个数据集上的测试结果 形如{'model1':{'data1':{'predict':[],'true':[]},.....},.......}
    data_name特定数据集名称
    
    return:
    result_dict 转换后的dict{'model1':{'predict':[],'true':[]},.......}
    """
    result_dict = {}
    for model_name in model_result_dict.keys():
        result_dict[model_name] = {'predict':model_result_dict.get(model_name).get(data_name).get('predict'),
                                    'true':model_result_dict.get(model_name).get(data_name).get('true')}
    return result_dict

def data_sample(n_folds=5,frac=0.2,X=None,y=None,groups=None,oob=True,random_state=0):
    """
    把数据集划分成多份用于模型训练
    n_folds:如果是int类型 那么就做bootstrap抽样 抽取n_folds份
            如果是是包含split函数的类 那么就调用其split函数 取出valid部分
    frac:抽取的样本比例 只有到n_folds是int的时候有效 值在0到1之间
    X: X数据 
    y: Y数据
    groups: 如果根据自定义的分组情况进行CV 那么就需要这个参数 比如LeaveOneGroupOut这个数据切分方法
    oob: 是否需要同时返回out of bag的index
    random_state:随机种子
    
    return:
    index_list n个index array组成的list
    """
    train_index_list = []
    oob_index_list = []
    num_samples = _num_samples(X)
    np.random.seed(random_state)
    if isinstance(n_folds,int):
        if frac is None:
            batch_size = round(num_samples/n_folds)
        elif frac >= 0 and frac <=1:
            batch_size = round(num_samples*frac)
        else:
            raise ValueError("expect frac is a int object between 0 and 1 but got {0}".format(frac))
        for i in range(n_folds):
            train_index = np.random.choice(num_samples,batch_size,replace=True)
            oob_index = [i for i in range(num_samples) if i not in train_index]
            train_index_list.append(train_index)
            oob_index_list.append(oob_index)
    elif hasattr(n_folds,'split'):
        for fold_n, (train_index, valid_index) in enumerate(n_folds.split(X,y,groups)):
            train_index_list.append(valid_index)
            oob_index_list.append(train_index)
    if oob:
        return train_index_list,oob_index_list
    else:
        return train_index_list


