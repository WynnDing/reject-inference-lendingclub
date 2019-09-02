# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer,MissingIndicator
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from pgy_utils import timecount,psi,check_unique


##Generate y value from y_related_var
def model_y_generator(orig_data, y_type, is_grey_good=False):
    """
    Generate y value from y_related_var

    :param raw_data: the orig data of model_data
    :type orig_data: pd.DataFrame
    :param y_type: the definition of y.
        'type_1': one month's loan order performance;
        'type_2': one month's first loan order performance;
    :type y_type: str
    :param is_grey_good: whether to consider grey samples as good.
        If true, consider grey account as good.
    :type is_grey_good: bool
    :return:
    :rtype:
    """
    raw_data = orig_data.copy()
    raw_data.reset_index(drop=True,inplace=True)
    # 按照好坏判断标准1
    if y_type == 'type_1':
        ##先将所有样本标记为-1；
        raw_data.loc[:, 'user_type'] = -1
        ##user_renzhengtype空代表用户在一个月内没有发起过借款，没有贷后表现，这个时候我们标记为-2；
        raw_data.loc[raw_data['user_renzhengtype'].isna(), 'user_type'] = -2
        ##data_1指代在一个月内发起过借款，并且订单已经完成的用户；
        data_1 = raw_data.loc[
            (raw_data['user_renzhengtype'] == 1) &
            (raw_data['status_5_count'] == 0) &
            (raw_data['status_1_count'] > 0),:]
        ##data_2指代在一个月内发起过借款并且不属于data_1的用户
        data_2 = raw_data.loc[((~raw_data['user_renzhengtype'].isna()) & (~raw_data.index.isin(data_1.index))),:]
        # 从data_1中取出好坏用户数据 灰色用户如果is_grey_good是True 那么就转成好用户
        data_1.loc[data_1['status_1_max_overdue_days'] == 0, 'user_type'] = 0
        data_1.loc[data_1['status_1_max_overdue_days'] > 3, 'user_type'] = 1
        if is_grey_good:
            data_1['user_type'] = data_1['user_type'].replace({-1: 0})
        # 从data_2中取出坏用户数据
        data_2.loc[
            (data_2['status_1_max_overdue_days'] > 3) |
            (data_2['status_5_max_overdue_days'] > 3), 'user_type'] = 1
        # 取出最后确定的标签
        raw_data.loc[data_1.index, 'user_type'] = data_1['user_type']
        raw_data.loc[data_2.index, 'user_type'] = data_2['user_type']
    # 按照好坏判断标准2
    elif y_type == 'type_2':
        raw_data.loc[:, 'user_type'] = -1
        raw_data.loc[raw_data['user_renzhengtype'].isna(), 'user_type'] = -2
        raw_data.loc[
            (raw_data['first_order_status'] == 1) &
            (raw_data['first_order_overdue_days'] == 0), 'user_type'] = 0
        raw_data.loc[
            (raw_data['first_order_status'] == 1) &
            (raw_data['first_order_overdue_days'] > 3), 'user_type'] = 1
        raw_data.loc[
            (raw_data['first_order_status'] == 5) &
            (raw_data['first_order_overdue_days'] > 3), 'user_type'] = 1

        if is_grey_good:
            raw_data.loc[
                (raw_data['first_order_status'] == 1) &
                (raw_data['first_order_overdue_days'] <= 3), 'user_type'] = 0
    return raw_data['user_type'].values



@timecount()
def data_simple_imputer(data_train,numeric_feature,category_feature,numeric_strategy='mean',category_strategy='most_frequent',data_test=None):
    '''
    使用DataFrameMapper进行简单的缺失值填补 指定类别型变量和连续型变量 并指定各自的填充策略
    data_train: 需要进行转换的训练集
    numeric_feature: 需要处理的数值型变量
    category_feature: 需要处理的类别型变量
    numeric_strategy: 连续型变量的填补策略 默认是均值
    category_strategy: 类别型变量的填补策略 默认是众数
    data_test: 需要进行转换的测试集 可以不给 不给就不会进行相应的转换
    
    return:
    X_train_imputed 添补完成的训练数据
    miss_transfer 训练好的DataFrameMapper类
    X_test_imputed 添补完成的测试数据 只有在给定测试数据的时候才会使用
    '''
    print('开始缺失值填充'.center(50, '='))
    ##从dict里面把特征list拿出来
    print('类别特征数',len(category_feature))
    print('数值特征数',len(numeric_feature))
    ##数值列和类别列用指定的方法填充
    miss_transfer = DataFrameMapper([
        (numeric_feature,[SimpleImputer(strategy=numeric_strategy)]),
        (category_feature,[SimpleImputer(strategy=category_strategy)])
    ])
    ##进行fit和transform
    X_train_imputed = miss_transfer.fit_transform(data_train[numeric_feature+category_feature])
    X_train_imputed = pd.DataFrame(X_train_imputed,columns=numeric_feature+category_feature)
    print('train_mapper完成:',X_train_imputed.shape)
    ##如果测试数据不为空 那么对测试数据进行transform 并返回
    if data_test is not None:
        X_test_imputed = miss_transfer.transform(data_test[numeric_feature+category_feature])
        X_test_imputed = pd.DataFrame(X_test_imputed,columns=numeric_feature+category_feature)
        return X_train_imputed,miss_transfer,X_test_imputed
    return X_train_imputed,miss_transfer

@timecount()
def data_missing_indicator(data_train,var_type_dict,data_test=None):
    '''
    进行特缺失值标记变量衍生
    data_train: 需要进行转换的训练集
    var_type_dict: 变量信息记录dict
    data_test: 需要进行转换的测试集 可以不给 不给就不会进行相应的转换
    
    return:
    data_train_completed 衍生完成的训练集
    var_type_dict 更新完的变量信息记录dict
    data_test_completed 衍生完成的测试集
    '''
    numeric_feature = var_type_dict.get('numeric_var',[])
    category_feature = var_type_dict.get('category_var',[])
    print('开始进行特缺失值标记变量衍生'.center(50, '='))
    ##从dict里面把特征list拿出来
    is_miss_feature = ['is_'+i+'_missing' for i in numeric_feature+category_feature]
    print('原始数据维度:',data_train.shape)
    print('新增数据维度:',len(is_miss_feature))
    check_unique(numeric_feature+is_miss_feature)
    ##数值列和类别列用指定的方法填充
    
    miss_indicator = MissingIndicator(features='all')
    data_train_completed = miss_indicator.fit_transform(data_train[numeric_feature+category_feature])
    data_train_completed = pd.concat([data_train,pd.DataFrame(data_train_completed,columns=is_miss_feature)],axis=1)
    print('变量衍生完成:',data_train_completed.shape)
    ##更新var_type_dict文件 全部加入到numeric_var当中
    var_type_dict['numeric_var'] = numeric_feature+is_miss_feature
    ##如果测试数据不为空 那么对测试数据进行transform 并返回
    if data_test is not None:
        data_test_completed = miss_indicator.transform(data_test[numeric_feature+category_feature])
        data_test_completed = pd.concat([data_test,pd.DataFrame(data_test_completed,columns=is_miss_feature)],axis=1)
        return data_train_completed,var_type_dict,data_test_completed
    return data_train_completed,var_type_dict


def PercentileThreshold(x,threshold=0.05,direction='both'):
    '''
    通过分位数来判断哪些值是异常值 可选双边异常值 单左异常值 单右异常值
    x: 一个一维的array 
    threshold: 区分异常值的分位数 0.05代表百分之5分位数
    direction: 可选both left right
    
    return:
    threshold_list 分别为左右的异常值cutoff point的值 list
    '''
    allowed_direction = ["both", "left", "right"]
    if direction not in allowed_direction:
        raise ValueError("Can only use these direction: {0} got direction={1}".format(allowed_direction, direction))
    threshold_percentile_dict ={'both':[threshold,1-threshold],'left':[threshold,1],'right':[0,threshold]} 
    threshold_percentile_list = threshold_percentile_dict.get(direction)
    threshold_list = np.array([np.nanpercentile(x,i*100) for i in threshold_percentile_list])
    return threshold_list

def isPercentileOuterlier(x,threshold_list):
    '''
    通过cutoff point来选出哪些为异常值
    x: 一个一维的array 
    threshold_list:分别为左右的异常值cutoff point的值 list
    
    return:
    一个一维的array 属于异常值的为True 其余为False
    '''
    return (x<threshold_list[0]) + (x>threshold_list[1])
    
    

class SimpleOutlierIndicator(BaseEstimator, TransformerMixin):
    '''
    一个自定义的简单通过分位数方法来判断outlier并打上标记的transformer 目前只适用于数值型的变量
    例：
    soi = SimpleOutlierIndicator()
    X_train = np.random.normal(1.75, 0.1, (10000, 40))
    soi.fit_transform(X_train)
    '''
    def __init__(self, threshold=0.05,direction='both'):         
        self.threshold = threshold
        self.direction = direction
    def fit(self, X, y=None):
        '''
            X必须是一个二维的array或者DataFrame
        '''
        if isinstance(X,pd.DataFrame):
            X = X.as_matrix()
        elif isinstance(X,np.ndarray):
            pass
        else:
            raise ValueError("paramter X got unexpected type")
        ##每次fit的时候都初始化
        self.threshold_list = []
        for i in range(X.shape[1]):
            self.threshold_list.append(PercentileThreshold(X[:,i],threshold=self.threshold,direction=self.direction))
        return self
    def transform(self, X, y=None):
        if isinstance(X,pd.DataFrame):
            X = X.as_matrix()
        elif isinstance(X,np.ndarray):
            pass
        else:
            raise ValueError("paramter X got unexpected type")
        result = np.zeros(X.shape, dtype = bool, order = 'C')
        for i in range(X.shape[1]):
            result[:,i] = isPercentileOuterlier(X[:,i],self.threshold_list[i])
        return result
    


