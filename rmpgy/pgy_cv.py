# -*- coding: utf-8 -*-
"""
Created on Wed May 15 12:09:45 2019

@author: sharpwhisper
"""

from sklearn.utils import indexable,check_consistent_length,check_array
from sklearn.utils.validation import _num_samples 
import numpy as np

class TimeGroupSplit():
    '''
    实现分组方式的时间序列数据拆分 比如按照时间顺序分组为1,2,3,4,5 组号小的表明是更早的数据
    '''
    def split(self, X, y=None, groups=None):
        '''
        实现分组方式的时间序列数据拆分 比如按照时间顺序分组为1,2,3,4,5 组号小的表明是更早的数据
        
        测试用例:
        X = np.array([[1,2],[3,4],[5,6],[7,8],[1,2],[3,4],[5,6],[7,8]])
        y = np.array([1,2,3,4,1,2,3,4])
        groups = np.array([1,2,1,2,3,4,3,4])
        tgs = TimeGroupSplit()
        for (train_index,test_index) in tgs.split(X,y,groups):
            print("%s %s"%(train_index,test_index))    
            
        输出结果为：
        [0 2] [1 3]
        [1 3] [4 6]
        [4 6] [5 7]
        '''
        ##把不能index的转成index 并比较index的长短是否一致
        X, y, groups = indexable(X, y, groups)
        ##确定groups是一个一维数组
        groups = check_array(groups, copy=True, ensure_2d=False, dtype=None)
        ##确定唯一组数 并排序
        unique_groups = np.unique(groups)
        unique_groups.sort()
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        for group_index in range(len(unique_groups)-1):
            yield (indices[groups==unique_groups[group_index]],
                   indices[groups==unique_groups[group_index+1]])



