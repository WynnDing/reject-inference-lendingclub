# -*- coding: utf-8 -*-
from datetime import datetime as dt
from functools import wraps
import numpy as np
import pandas as pd
from sklearn.utils.validation import _num_samples 


##a docorate function to estimate time consuming of the target fun 
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

def is_monotonic(x):
    '''
    判断序列是否单调
    x list或者一维得np.array或者pd.Series
    '''
    dx = np.diff(x)
    return np.all(dx < 0).astype(int) or np.all(dx > 0).astype(int)

def check_unique(x):
    '''
    判断序列是否存在重复值
    x list或者一维得np.array或者pd.Series
    '''
    if len(np.unique(x))!=len(x):
        return False
    else:
        return True
    

def check_non_intersect(x,y):
    '''
    判断序列x和序列y是否存在交集
    x list或者一维np.array或者pd.Series
    y list或者一维得np.array或者pd.Series
    '''
    if len(set(x) & set(y)) != 0:
        print("存在交集:%s"%(set(x) & set(y)))
        return False
    else:
        return True

def dict_reverse(orig_dict):
    '''
    把一个map中的key和value反转，返回的map以之前的value作为key，并且每个value对应之前的一系列key组成的list
    '''
    reverse_dict = {}
    for key in orig_dict.keys():
        value = orig_dict.get(key)
        if value not in reverse_dict.keys():
            reverse_dict[value] = [key]
        else:
            temp_list = reverse_dict.get(value)
            temp_list.append(key)
            reverse_dict[value] = temp_list
    return reverse_dict


def psi(s1,s2):
    """
    用于计算psi s1和s2的长度必须相等
    s1: 对比序列1分箱之后每一箱的数据占比
    s2: 对比序列2分箱之后每一箱的数据占比
    """
    psi = 0
    s1 = list(s1)
    s2 = list(s2)
    if len(s1) != len(s2):
        print('序列s1和s2长度不等 请检查!')
        return None
    for i in range(len(s1)):
        ##处理下部分箱为0的情况
        if s2[i] == 0:
            s2[i]=0.000001
        if s1[i] == 0:
            s1[i]=0.000001
        p = ((s2[i]-s1[i])*(np.log(s2[i]/s1[i])))
        psi = psi+p
    return psi








