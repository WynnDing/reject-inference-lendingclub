# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:51:09 2019

@author: sharpwhisper
"""
import pandas as pd
import numpy as np
from pgy_utils import is_monotonic,psi,timecount,check_unique,dict_reverse
import warnings
warnings.filterwarnings("ignore")


"""
分箱、WOE转换以及IV相关函数
"""
##---------------------------------无监督分箱--------------------------------------------
def interpolate_binning(data,var,special_attribute=[]):
    '''
    用于插值分箱（每两个值之间作为一个分箱节点） 返回分箱节点 不允许存在缺失值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的插值分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    value_list = list(binning_series.value_counts().sort_index().index)
    cp = [(value_list[i]+value_list[i+1])/2 for i in np.arange(len(value_list)-1)]
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

def quantile_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于等频分箱 返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    
    '''
    print('正在进行变量{0}的等频分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##这里用1:-1的原因是10分箱只需要9个cut off point就可以了
    cp = [binning_series.quantile(i) for i in np.linspace(0,1,max_interval+1)[1:-1]]
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

def distance_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于等距分箱返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的等距分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##这里用1:-1的原因是10分箱只需要9个cut off point就可以了
    cp = list(np.linspace(binning_series.min(),binning_series.max(),max_interval+1,endpoint=True)[1:-1])
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

def mix_binning(data, var, max_interval=10, special_attribute=[]):
    '''
    用于混合分箱返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    混合分箱的存在是为了防止异常值的存在对等距分箱的影响 在头尾进行等频率的分箱 然后剩下的部分用等距分箱
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的混合分箱'.format(var))
    ##从DataFrame里面取出对应列的Series 并做好排序工作
    binning_series = data[var].loc[~data[var].isin(special_attribute)].sort_values()
    ##判断是否存在缺失值
    if np.sum(binning_series.isna())>0:
        raise ValueError("detect nan values in {0}".format(var))
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_series.value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    ##混合分箱
    quantile_cp = [binning_series.quantile(i) for i in np.linspace(0,1,max_interval+1)[1:-1]]
    distance_cp = list(np.linspace(quantile_cp[0],quantile_cp[-1],max_interval-1,endpoint=True)[1:-1])
    cp = [quantile_cp[0]] + distance_cp + [quantile_cp[-1]]
    ##判断分箱点是否存在重复值
    if not check_unique(cp):
        print("quantile cut off points for {0} with {1} bins is not unique, need extra operation".format(var,max_interval))
        cp = sorted(list(set(cp)))
    return cp

##---------------------------------有监督决策树分箱--------------------------------------------
from sklearn.tree import DecisionTreeClassifier
def tree_binning(data, var, label, special_attribute=[],treeClassifier=DecisionTreeClassifier()):
    """
    用于决策树分箱 不允许存在缺失值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    treeClassifier: sklearn的DecisionTreeClassifier类或者有类似功能的其他类
    
    return:
    cp: 分箱切分的节点
    """
    print('正在进行变量{0}的决策树分箱'.format(var))
    ##从DataFrame里面取出对应的数据
    binning_data = data[[var, label]].loc[~data[var].isin(special_attribute)]
    ##判断是否存在缺失值
    if (np.sum(binning_data[var].isna())>0) or (np.sum(binning_data[label].isna())>0):
        raise ValueError("detect nan values in {0}".format([var,label]))
    ##进行决策树拟合
    treeClassifier.fit(X=binning_data[var].values.reshape(-1, 1),y=binning_data[label].values.reshape(-1, 1))
    cp = sorted(treeClassifier.tree_.threshold[treeClassifier.tree_.threshold != -2])
    ##处理没有找到任何可能分箱的情况
    if len(cp) == 0:
        raise ValueError("detect empty cp for {0} in tree_binning".format([var,label]))
    return cp

##---------------------------------有监督卡方分箱--------------------------------------------
#-----------------辅助函数1 初始化数据分箱-----------
def SplitData(df, col, numOfSplit, special_attribute=[]):
    """
    在原数据集上增加一列，把原始细粒度的col重新划分成粗粒度的值，便于分箱中的合并处理
    :param df: 按照col排序后的数据集
    :param col: 待分箱的变量
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外
    :return: 
    splitPoint： 初始化数据分箱的节点
    """
    df2 = df.copy()
    if len(special_attribute) > 0:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = int(N / numOfSplit)
    splitPointIndex = [i*n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))

    aa = pd.Series(splitPoint)
    if (aa[0] == 0.0) & (aa.shape[0] == 1):
        numOfSplit = 1000
        n = int(N / numOfSplit)
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(df2[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
    else:
        pass
    return splitPoint

#-----------------辅助函数2 数据在不同分组下的节点映射----------
def AssignGroup(x, bin):
    """
    :return: 数值x在区间映射下的结果。例如，x=2，bin=[0,3,5], 由于0<x<3,x映射成3
    """
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]

#--------------------辅助函数3 判断每一个分组里面是否有好坏样本------------
def BinBadRate(df, col, target, grantRateIndicator=0):
    """
    用于计算col中数值对应的坏样本的占比情况
    :param df: 需要计算好坏比率的数据集
    :param col: 需要计算好坏比率的特征
    :param target: 好坏标签
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回
    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad / x.total, axis=1)
    dicts = dict(zip(regroup[col], regroup['bad_rate']))
    if grantRateIndicator == 0:
        return dicts, regroup
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return dicts, regroup, overallRate 

#----------------------辅助函数4 卡方值的计算-----------------
def Chi2(df, total_col, bad_col):
    """
    用于计算初始化分箱后相邻分箱的卡方值
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :return: 
    chi2：卡方值
    """
    df2 = df.copy()
    # 求出df中，总体的坏样本率和好样本率
    badRate = sum(df2[bad_col]) * 1.0 / sum(df2[total_col])
    # 当全部样本只有好或者坏样本时，卡方值为0
    if badRate in [0, 1]:
        return 0
    df2['good'] = df2.apply(lambda x: x[total_col] - x[bad_col], axis=1)
    goodRate = sum(df2['good']) * 1.0 / sum(df2[total_col])
    # 期望坏（好）样本个数＝全部样本个数*平均坏（好）样本占比
    df2['badExpected'] = df[total_col].apply(lambda x: x * badRate)
    df2['goodExpected'] = df[total_col].apply(lambda x: x * goodRate)
    badCombined = zip(df2['badExpected'], df2[bad_col])
    goodCombined = zip(df2['goodExpected'], df2['good'])
    badChi = [(i[0]-i[1])**2 / i[0] for i in badCombined]
    goodChi = [(i[0] - i[1]) ** 2 / i[0] for i in goodCombined]
    chi2 = sum(badChi) + sum(goodChi)
    return chi2

#--------------辅助函数5 特征数值对应的分箱的组别------------------------
def AssignBin(x, cutOffPoints,special_attribute=[]):
    """
    x取值根据分箱的节点，对应到相应的第几个中例如, cutOffPoints = [10,20,30], 对于 x = 7, 返回 Bin 0；对于x=23，返回Bin 2； 对于x = 35, return Bin 3。
    对于特殊值，返回的序列数前加"-"
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
   
    """
    cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
    numBin = len(cutOffPoints2)
    # print('x = ', x)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x <= cutOffPoints2[0]:
        return 'Bin 0'
    elif x > cutOffPoints2[-1]:
        return 'Bin {}'.format(numBin)
    else:
        for ii in range(0, numBin):
            if cutOffPoints2[ii] < x <= cutOffPoints2[ii+1]:
                return 'Bin {}'.format(ii+1)     
 
#------------------辅助函数6 根据好坏样本对分箱进行合并-----------------
                
def Bad_Rate_Merge(df, col, cutOffPoints, target):
    """
    根据各个分箱中的坏样本的占比情况，对分箱中坏用户占比小于阈值的箱进行上下箱的合并
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param col: 计算好坏样本占比的列
    :param cutOffPoints: 分箱节点
    :param target: 目标变量
    :return:
	cutOffPoints 分箱节点
    """
    while 1:
        df2 = df.copy()
        # print('cutOffPoints', cutOffPoints)
        df2['temp_Bin'] = df2[col].apply(lambda x: AssignBin(x, cutOffPoints))
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        # 找出全部为好／坏样本的箱
        indexForBad01 = regroup[
            regroup['bad_rate'].isin([0, 1])]['temp_Bin'].tolist()
        if len(indexForBad01) == 0:
            return cutOffPoints
        bin = indexForBad01[0]
        # 如果是最后一箱，则需要和上一个箱进行合并，
        # 也就意味着分裂点cutOffPoints中的最后一个需要移除
        if bin == max(regroup['temp_Bin']):
            cutOffPoints = cutOffPoints[:-1]
            if len(cutOffPoints) == 0:
                return np.nan
        # 如果是第一箱，则需要和下一个箱进行合并，
        # 也就意味着分裂点cutOffPoints中的第一个需要移除
        elif bin == min(regroup['temp_Bin']):
            cutOffPoints = cutOffPoints[1:]
            if len(cutOffPoints) == 0:
                return np.nan
        # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
        else:
            # 和前一箱进行合并，并且计算卡方值
            currentIndex = list(regroup['temp_Bin']).index(bin)
            prevIndex = list(regroup['temp_Bin'])[currentIndex - 1]
            df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
            (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
            chisq1 = Chi2(df2b, 'total', 'bad')
            # 和后一箱进行合并，并且计算卡方值
            laterIndex = list(regroup['temp_Bin'])[currentIndex + 1]
            df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
            (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
            chisq2 = Chi2(df2b, 'total', 'bad')
            if chisq1 < chisq2:
                cutOffPoints.remove(cutOffPoints[currentIndex - 1])
            else:
                cutOffPoints.remove(cutOffPoints[currentIndex])
        # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
        df2['temp_Bin'] = df2[col].apply(lambda x: AssignBin(x, cutOffPoints))
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                    max(binBadRate.values())]
        if minBadRate > 0 and maxBadRate < 1:
            break
    return cutOffPoints

#---------------------------助攻函数1完整的卡方分箱函数--------------
def ChiMerge(
        df, col, target, max_interval, special_attribute, minBinPcnt=0
):
    """
	完整的卡方分箱函数
    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数。如果原始属性的取值个数低于该参数，不执行这段函数
    :param special_attribute: 不参与分箱的属性取值
    :param minBinPcnt：最小箱的占比，默认为0
    :return: 
	cutOffPoints 分箱节点
    """
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:
        # 如果原始属性的取值个数低于max_interval，不执行这段函数
        print("   The number of original levels for {} is less than or equal "
              "to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute) >= 1:
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))

        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))###这个地方可以直接将原来的分100组改成上面卡方分箱的结果
            if len(df2['temp'].unique()) == 1:
                return np.nan
        else:
            # print(df2.shape)
            df2['temp'] = df2[col]
            # print(df2.shape)
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(
            df2, 'temp', target, grantRateIndicator=1
        )
        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到：
        # 1，最终分裂出来的分箱数<＝预设的最大分箱数
        # 2，每箱的占比不低于预设值（可选）
        # 3，每箱同时包含好坏样本
        # 如果有特殊属性，那么最终分裂出来的分箱数＝预设的最大分箱数－特殊属性的个数
        split_intervals = max_interval - len(special_attribute)
        if split_intervals == 1:
            return np.nan
        while len(groupIntervals) > split_intervals:
            # 终止条件: 当前分箱数＝预设的分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值。具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals)-1):
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad')
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + \
                                             groupIntervals[best_comnbined+1]
            # 当将最优的相邻的两个变量合并在一起后，需要从原来的列表中将其移除。
            # 例如，将[3,4,5] 与[6,7]合并成[3,4,5,6,7]后，需要将[3,4,5]
            # 与[6,7]移除，保留[3,4,5,6,7]
            groupIntervals.remove(groupIntervals[best_comnbined+1])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        # 检查是否有箱没有好或者坏样本。
        # 如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        df2['temp_Bin'] = df2['temp'].apply(
            lambda x: AssignBin(x, cutOffPoints)
        )
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                    max(binBadRate.values())]
        if minBadRate == 0 or maxBadRate == 1:
            cutOffPoints = Bad_Rate_Merge(df, col, cutOffPoints, target)

        # 需要检查分箱后的最小占比
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(
                lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            valueCounts = groupedvalues.value_counts().to_frame()
            N = sum(valueCounts['temp'])
            valueCounts['pcnt'] = valueCounts['temp'].apply(
                lambda x: x * 1.0 / N)
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                # 找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'
                                              ] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，
                # 也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，
                # 也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，
                # 依据是较小的卡方值
                else:
                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valueCounts.index).index(
                        indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin(
                        [prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b, 'total', 'bad')
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin(
                        [laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    chisq2 = Chi2(df2b, 'total', 'bad')
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])
                groupedvalues = df2['temp'].apply(
                    lambda x: AssignBin(x, cutOffPoints))
                df2['temp_Bin'] = groupedvalues
                valueCounts = groupedvalues.value_counts().to_frame()
                valueCounts['pcnt'] = valueCounts['temp'].apply(
                    lambda x: x * 1.0 / N)
                valueCounts = valueCounts.sort_index()
                minPcnt = min(valueCounts['pcnt'])
        if cutOffPoints is np.nan:
            return np.nan
        return cutOffPoints

##--------------------------------------------------------------分箱单调性判断并合并--------------------------------------------------
#----------------------单调性辅助函数1 寻找不单调的位置--------------
def FeatureMonotone(x):
    """
    :return: 返回序列x中有几个元素不满足单调性，以及这些元素的位置。
    例如，x=[1,3,2,5], 元素3比前后两个元素都大，不满足单调性；元素2比前后两个元素都小，也不满足单调性。
    故返回的不满足单调性的元素个数为2，位置为1和2.
    """
    monotone = [(x[i] < x[i+1]) and (x[i] < x[i-1]) or (x[i]>x[i+1]) and
                (x[i] > x[i-1]) for i in range(1, len(x)-1)]
    index_of_nonmonotone = [i+1 for i in range(len(monotone)) if monotone[i]]
    return {'count_of_nonmonotone': monotone.count(True),
            'index_of_nonmonotone': index_of_nonmonotone}

#---------------------单调性辅助函数2  判断分组内是否单调------------
def BadRateMonotone(df, sortByVar, target, special_attribute=[]):
    """
    判断某变量的坏样本率是否单调
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    """
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateNotMonotone = FeatureMonotone(badRate)['count_of_nonmonotone']
    if badRateNotMonotone > 0:
        return False
    else:
        return True

#--------------------单调性辅助函数3 不单调的分组进行合并--------------------
def Monotone_Merge(df, target, col):
    """
    :return:将数据集df中，不满足坏样本率单调性的变量col进行合并，使得合并后的新的变量中，坏样本率单调，输出合并方案。
    例如，col=[Bin 0, Bin 1, Bin 2, Bin 3, Bin 4]是不满足坏样本率单调性的。合并后的col是：
    [Bin 0&Bin 1, Bin 2, Bin 3, Bin 4].
    合并只能在相邻的箱中进行。
    迭代地寻找最优合并方案。每一步迭代时，都尝试将所有非单调的箱进行合并，每一次尝试的合并都是跟前后箱进行合并再做比较
    """
    def MergeMatrix(m, i, j, k):
        """
        :param m: 需要合并行的矩阵
        :param i, j: 合并第i和j行
        :param k: 删除第k行
        :return: 合并后的矩阵
        """
        m[i, :] = m[i, :] + m[j, :]
        m = np.delete(m, k, axis=0)
        return m

    def Merge_adjacent_Rows(
            i, bad_by_bin_current, bins_list_current,
            not_monotone_count_current
    ):
        """
        :param i: 需要将第i行与前、后的行分别进行合并，比较哪种合并方案最佳。判断准则是，合并后非单调性程度减轻，且更加均匀
        :param bad_by_bin_current:合并前的分箱矩阵，包括每一箱的样本个数、坏样本个数和坏样本率
        :param bins_list_current: 合并前的分箱方案
        :param not_monotone_count_current:合并前的非单调性元素个数
        :return:分箱后的分箱矩阵、分箱方案、非单调性元素个数和衡量均匀性的指标balance
        """
        i_prev = i - 1
        i_next = i + 1
        bins_list = bins_list_current.copy()
        bad_by_bin = bad_by_bin_current.copy()
        not_monotone_count = not_monotone_count_current
        #合并方案a：将第i箱与前一箱进行合并
        bad_by_bin2a = MergeMatrix(bad_by_bin.copy(), i_prev, i, i)
        bad_by_bin2a[i_prev, -1] = bad_by_bin2a[i_prev, -2] / bad_by_bin2a[
            i_prev, -3]
        not_monotone_count2a = FeatureMonotone(bad_by_bin2a[:, -1]
                                               )['count_of_nonmonotone']
        # 合并方案b：将第i行与后一行进行合并
        bad_by_bin2b = MergeMatrix(bad_by_bin.copy(), i, i_next, i_next)
        bad_by_bin2b[i, -1] = bad_by_bin2b[i, -2] / bad_by_bin2b[i, -3]
        not_monotone_count2b = FeatureMonotone(bad_by_bin2b[:, -1])[
            'count_of_nonmonotone']
        balance = ((bad_by_bin[:, 1] / N).T * (bad_by_bin[:, 1] / N))[0, 0]
        balance_a = ((bad_by_bin2a[:, 1] / N).T * (bad_by_bin2a[:, 1] / N)
                     )[0, 0]
        balance_b = ((bad_by_bin2b[:, 1] / N).T * (bad_by_bin2b[:, 1] / N)
                     )[0, 0]
        # 满足下述2种情况时返回方案a：（1）方案a能减轻非单调性而方案b不能；
        # （2）方案a和b都能减轻非单调性，但是方案a的样本均匀性优于方案b
        if (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b >= not_monotone_count_current) or \
                (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current) and \
                (balance_a < balance_b):
            bins_list[i_prev] = bins_list[i_prev] + bins_list[i]
            bins_list.remove(bins_list[i])
            bad_by_bin = bad_by_bin2a
            not_monotone_count = not_monotone_count2a
            balance = balance_a
        # 同样地，满足下述2种情况时返回方案b：
        # （1）方案b能减轻非单调性而方案a不能；
        # （2）方案a和b都能减轻非单调性，但是方案b的样本均匀性优于方案a
        elif (not_monotone_count2a >= not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current)or \
                (not_monotone_count2a < not_monotone_count_current) and \
                (not_monotone_count2b < not_monotone_count_current) and \
                (balance_a > balance_b):
            bins_list[i] = bins_list[i] + bins_list[i_next]
            bins_list.remove(bins_list[i_next])
            bad_by_bin = bad_by_bin2b
            not_monotone_count = not_monotone_count2b
            balance = balance_b
        # 如果方案a和b都不能减轻非单调性，返回均匀性更优的合并方案
        else:
            if balance_a< balance_b:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
            else:
                bins_list[i] = bins_list[i] + bins_list[i_next]
                bins_list.remove(bins_list[i_next])
                bad_by_bin = bad_by_bin2b
                not_monotone_count = not_monotone_count2b
                balance = balance_b
        return {'bins_list': bins_list, 'bad_by_bin': bad_by_bin,
                'not_monotone_count': not_monotone_count,
                'balance': balance}

    N = df.shape[0]
    [badrate_bin, bad_by_bin] = BinBadRate(df, col, target)
    bins = list(bad_by_bin[col])
    bins_list = [[i] for i in bins]
    badRate = sorted(badrate_bin.items(), key=lambda x: x[0])
    badRate = [i[1] for i in badRate]
    not_monotone_count, not_monotone_position = \
        FeatureMonotone(badRate)['count_of_nonmonotone'], \
        FeatureMonotone(badRate)['index_of_nonmonotone']
    # 迭代地寻找最优合并方案，终止条件是:当前的坏样本率已经单调，或者当前只有2箱
    while not_monotone_count > 0 and len(bins_list) > 2:
        # 当非单调的箱的个数超过1个时，每一次迭代中都尝试每一个箱的最优合并方案
        all_possible_merging = []
        for i in not_monotone_position:
            merge_adjacent_rows = Merge_adjacent_Rows(
                i, np.mat(bad_by_bin), bins_list, not_monotone_count)
            all_possible_merging.append(merge_adjacent_rows)
        balance_list = [i['balance'] for i in all_possible_merging]
        not_monotone_count_new = [
            i['not_monotone_count'] for i in all_possible_merging]
        # 如果所有的合并方案都不能减轻当前的非单调性，就选择更加均匀的合并方案
        if min(not_monotone_count_new) >= not_monotone_count:
            best_merging_position = balance_list.index(min(balance_list))
        # 如果有多个合并方案都能减轻当前的非单调性，也选择更加均匀的合并方案
        else:
            better_merging_index = [
                i for i in range(len(not_monotone_count_new))
                if not_monotone_count_new[i] < not_monotone_count
            ]
            better_balance = [balance_list[i] for i in better_merging_index]
            best_balance_index = better_balance.index(min(better_balance))
            best_merging_position = better_merging_index[best_balance_index]
        bins_list = all_possible_merging[best_merging_position]['bins_list']
        bad_by_bin = all_possible_merging[best_merging_position]['bad_by_bin']
        not_monotone_count = all_possible_merging[best_merging_position
        ]['not_monotone_count']
        not_monotone_position = FeatureMonotone(bad_by_bin[:, 3]
                                                )['index_of_nonmonotone']
    return bins_list

#--------------------卡方分箱函数的最外层调用函数--------------------
def chi_binning(data, var, label, max_interval=10, special_attribute=[]):
    '''
    用于等频分箱 返回分箱节点 不允许存在缺失值 不同值的个数一定要超过max_interval的值
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    max_interval: 分箱的组数 int
    special_attribute: 在切分数据集的时候，某些特殊值需要排除在外 list
    
    return:
    cp: 分箱切分的节点
    '''
    print('正在进行变量{0}的卡方分箱'.format(var))
    ##从DataFrame里面取出对应的数据
    binning_data = data[[var, label]].loc[~data[var].isin(special_attribute)]
    
    ##判断是否存在缺失值
    if (np.sum(binning_data[var].isna())>0) or (np.sum(binning_data[label].isna())>0):
        raise ValueError("detect nan values in {0}".format([var,label]))
       
    ##判断不同值的个数是否满足条件
    different_value_nums = len(binning_data[var].value_counts())
    if different_value_nums < max_interval:
        raise ValueError("value_counts for {0} is {1}, less than max_interval {2}".format(var,different_value_nums,max_interval))
    
    ##进行卡方分箱
    cp = ChiMerge(data, var, label, max_interval=max_interval, special_attribute=special_attribute)
    if (cp is np.nan) or not cp:
        raise ValueError("detect empty cp for {0} in chi_binning".format([var,label]))
        
    return cp

#----------------------单调性助攻函数 判断分组是否单调并进行合并---------------------
def cutpoint_BRM(data, var,label,cp):
    '''
    用于检验每一个分箱中的坏样本分布是否单调，如果不单调需要将不单调的分箱进行上下合并
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    cp: 分箱切分的节点 list

    return:
	cp: 经过单调性检验合并后的分箱节点
    '''
    print('正在进行变量{0}的分箱单调性合并工作'.format(var))
    
    data = data.copy()
    special_attribute = []
    var_cutoff = dict()
    col1 = str(var) + '_Bin'
    cp =sorted(list(cp))

    data[col1] = data[var].map(
        lambda x: AssignBin(
            x, cp, special_attribute=special_attribute
        )
    )

    (binBadRate, regroup) = BinBadRate(
        data, col1, label
    )
    [minBadRate, maxBadRate] = [min(binBadRate.values()),
                                max(binBadRate.values())]
    if minBadRate == 0 or maxBadRate == 1:
        if len(binBadRate) == 2:
            return np.nan
        else:
            cp = Bad_Rate_Merge(data, var,cp,label)                                                        
            return cp
    data[col1] = data[var].map(
        lambda x: AssignBin(
            x, cp, special_attribute=special_attribute
        )
    )
    var_cutoff[var] = cp
    # print(var_cutoff)
    BRM = BadRateMonotone(
        data, col1, label, special_attribute=special_attribute
    )
    if not BRM:
        bin_merged = Monotone_Merge(data, label, col1)
        removed_index = []
        for bin in bin_merged:
            if len(bin) > 1:
                indices = [int(b.replace('Bin ', '')) for b in bin]
                removed_index = removed_index + indices[0:-1]
        removed_point = [cp[k] for k in removed_index]
        for p in removed_point:
            cp.remove(p)
        var_cutoff[var] = cp
        data[col1] = data[var].map(
            lambda x: AssignBin(
                x, cp, special_attribute=special_attribute
            )
        )
    return cp


##---------------------------------根据分箱后的节点对特征进行woe转换并计算iv值--------------------------------------------
import statsmodels.api as sm

def numeric_var_binning_with_bins(data, var, label, bins, woe_shift=0.000001):
    '''
    根据用户给出的分箱节点进行连续型单变量进行分箱操作 并进行woe转换和iv计算
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    bins: 分箱节点 list
    woe_shift: 为防止极端情况出现 在计算woe的时候加上一个小常数
    
    
    return: 
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    IV: 最终的IV值 float
    bins: 分箱节点 list
    '''
    print('正在进行数值型变量{0}的IV计算工作'.format(var))
    ##拷贝一份数据出来 防止篡改原本的数据
    df = data[[var, label]].copy()
    ##进行实际的分箱工作 并计算odds woe IV等指标
    df['bins'] = pd.cut(x=df[var], bins=bins)
    bins_df = pd.crosstab(df['bins'], df[label])
    bins_df.columns = ['num_0', 'num_1']
    bins_df['num_01'] = bins_df.apply(lambda x: sum(x), axis=1)
    ##给出每个组内包含的值 为了跟类别型变量统一
    bins_df['contains_values'] = bins_df.index
    ##调整下列的顺序
    bins_df = bins_df[['contains_values','num_0','num_1','num_01']]
    
    bins_df['pct_0_row'] = bins_df['num_0'] / bins_df['num_01']
    bins_df['pct_1_row'] = bins_df['num_1'] / bins_df['num_01']
    bins_df['pct_0_col'] = bins_df['num_0'] / bins_df['num_0'].sum()
    bins_df['pct_1_col'] = bins_df['num_1'] / bins_df['num_1'].sum()
    
    ##1作为坏用户标签是我们的响应变量 woe越大说明坏用户越多
    bins_df['odds'] = bins_df['pct_1_row'] / bins_df['pct_0_row']
    bins_df['woe'] = np.log(bins_df['pct_1_col'] / bins_df['pct_0_col'])
    bins_df['miv'] = (bins_df['pct_1_col'] - bins_df['pct_0_col']) * bins_df['woe']
    ##由于容易存在小类别样本中只有一种的情况 这里对np.inf做相应的处理 暂时以0考虑
    bins_df['woe'] = bins_df['woe'].replace({np.inf:0,-np.inf:0})
    bins_df['miv'] = bins_df['miv'].replace({np.inf:0,-np.inf:0})
    ##计算IV
    IV = bins_df['miv'].sum()
    bins_df['IV'] = [IV] * bins_df.shape[0]
    ##给出是否单调的判断
    bins_df['is_monotonic'] = [is_monotonic(bins_df['woe'].values)] * bins_df.shape[0]
    ##TODO 给出更好的方向指标
    return bins_df, IV, bins


def numeric_var_binning(data, var, label, max_interval=10,method='Equifrequency',BRM=False,special_attribute=[],
                        tree_params={"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.10,"random_state":0},
                        woe_shift=0.000001):
    '''
    根据用户指定的方式对连续型单变量进行分箱操作 并进行woe转换和iv计算 special_attribute一般用于缺失值 如果使用一个较小的负数进行填补 那么就把其加入到其中单独分箱 否则不单独处理
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    max_interval: 最大分箱数 对决策树分箱不起作用 int
    method: "ChiMerge"(卡方分箱);"DecisionTree"（决策树分箱);"Equifrequency"(等频分箱);"Equidistance"(等距分箱);"Mix"(混合分箱);"Interpolate"(插值分箱)
    BRM: 是否需要根据分箱的单调性进行分箱合并，默认False
    special_attribute：不需要进行分箱的特殊点(单独一个箱)
    tree_params:决策树分箱所用决策树的参数
    woe_shift:为防止极端情况出现 在计算woe的时候加上一个小常数
    
    return: 
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    IV: 最终的IV值 float
    bins: 分箱节点 list
    '''
    allowed_method = ["ChiMerge", "DecisionTree", "Equifrequency","Equidistance","Mix",'Interpolate']
    if method not in allowed_method:
        raise ValueError("Can only use these method: {0} got method={1}".format(allowed_method, method))
    
    ##拷贝一份数据出来 防止篡改原本的数据
    df = data[[var, label]].copy()
    if method=='ChiMerge':
        cp=chi_binning(df, var, label, max_interval, special_attribute)
    elif method=='DecisionTree':
        cp=tree_binning(df, var, label, special_attribute,DecisionTreeClassifier(**tree_params))
    elif method=='Equifrequency':
        cp=quantile_binning(df, var, max_interval, special_attribute)
    elif method=='Equidistance':
        cp=distance_binning(df, var, max_interval, special_attribute)
    elif method=='Mix':
        cp=mix_binning(df, var, max_interval, special_attribute)
    elif method=='Interpolate':
        cp=interpolate_binning(df,var,special_attribute)
    
    ##是否需要进行单调性合并
    if BRM:
        cp=cutpoint_BRM(data, var, label, cp)
    
    ##检查special attribute是否是为空或者为一个足够小的负数
    if len(special_attribute) > 1 or ( len(special_attribute) == 1 and special_attribute[0] > df[var].min()):
        raise ValueError("special_attribute for {0} is {1}, not a empty list or a one element list contains the smallest value".format(var,special_attribute))
    ##合并最终的分箱节点 
    bins = sorted([-np.inf]+special_attribute+cp+[np.inf])
    ##根据分箱节点计算woe、iv等
    bins_df, IV, bins = numeric_var_binning_with_bins(df, var, label, bins, woe_shift)
    return bins_df, IV, bins


    

def numeric_var_woe_trans(data, var, bins_df, bins):
    '''
	根据已经完成的分箱操作 对单个连续变量进行相应的woe转化工作
    data: 数据源 DataFrame
    var: woe转化目标变量 str
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    bins: 分箱节点 list
    
    return:
    woe_array: 转换完成后每个样本对应的woe值 array
	'''
    df = data[[var]].copy()
    ##进行数据转换的对应dict
    trans_dict = dict(zip(bins_df.index.tolist(), bins_df['woe'].values))
    ##进行分箱操作并转换woe 这里逻辑必须跟cut_bins_var函数一致
    df['bins'] = pd.cut(x=df[var], bins=bins)
    df['woe'] = df['bins'].replace(trans_dict)
    return df['woe'].values



def category_var_bins_combine(data, var, label, max_interval=10, method='default',special_attribute=[],\
                              tree_params={"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.01,"random_state":0}):
    '''
    用于类别型特征基于各类坏样本的占比情况进行降序排序给出新的组号，并基于此进行新一轮的分箱重组，返回原本的类别对应的最终组号
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    max_interval: 最大分箱数 在method为"default"时不起作用 int 
    method: "default"(按照降序排序后输出对应关系); "ChiMerge"(按照降序排序后进行卡方分箱并输出对应关系)；"DecisionTree"(按照降序排序后进行决策树分箱并输出对应关系)
    special_attribute：不需要进行分箱的特殊点(单独一个箱)   
    tree_params: 决策树分箱所用决策树的参数
    
    return：
    bins 类别特征值及对应的新组号
    '''
    print('正在进行类别型变量{0}的重新分组的工作'.format(var))
    allowed_method = ["default","ChiMerge","DecisionTree"]
    if method not in allowed_method:
        raise ValueError("Can only use these method: {0} got method={1}".format(allowed_method, method))
    ##把不需要进行分箱的样本从中拿掉
    df=data[[var,label]][~data[var].isin(special_attribute)].copy()
    ##计算原本每个类别的坏用户占比 
    bins_df = pd.crosstab(df[var], df[label])
    bins_df.columns = ['num_0', 'num_1']
    bins_df['num_01'] = bins_df.apply(lambda x: sum(x), axis=1)
    bins_df['pct_0_row'] = bins_df['num_0'] / bins_df['num_01']
    bins_df['pct_1_row'] = bins_df['num_1'] / bins_df['num_01']
    ##按照降序排序 给出新的组号 记录对应关系
    var_order=list(bins_df.sort_values(by="pct_1_row" , ascending=False).index)
    stage1_bins_dict={}
    for i in var_order:
        stage1_bins_dict[i]=var_order.index(i)
    ##给出新组号的列
    df[var+'_stage1_bin'] = df[var].replace(stage1_bins_dict)
    ##按照不同的method进行处理
    stage2_bins_dict={}
    if method == 'default':
        ##对于default方法 只需要把降序的对应关系稍加处理 把新分组的名称改成Bin_x这样即可
        for key in stage1_bins_dict.keys():
            stage2_bins_dict[key] =  "Bin_"+"%03d"%(stage1_bins_dict.get(key))
    elif method == 'ChiMerge':
        ##对于ChiMerge方法 要把降序后的新分组当成变量传入到卡方分箱当中 然后把划分点拿出来对降序后的组号进行再分组 然后建立原本类别到最终分组的对应关系
        cp=chi_binning(df, var+'_stage1_bin', label, max_interval)
        stage2_bins=sorted([-np.inf]+cp+[np.inf])
        ##建立降序分组的对应关系 并把卡方分组的对应关系附加上去
        stage1_bin_df = pd.DataFrame({'stage1_bin':np.arange(len(stage1_bins_dict))})
        stage1_bin_df['stage2_bin'] = pd.cut(x=stage1_bin_df['stage1_bin'], bins=stage2_bins)
        stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin']
        index = 0
        for i in stage1_bin_df['stage2_bin'].value_counts().sort_index().index:
            stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin_final'].replace({i:'Bin_'+"%03d"%(index)})
            index += 1
        for i in stage1_bins_dict.keys():
            stage2_bins_dict[i] = stage1_bin_df['stage2_bin_final'][stage1_bin_df['stage1_bin'] == stage1_bins_dict.get(i)].values[0]
    else:
        cp=tree_binning(df, var+'_stage1_bin', label,treeClassifier = DecisionTreeClassifier(**tree_params))
        stage2_bins=sorted([-np.inf]+cp+[np.inf])
        ##建立降序分组的对应关系 并把卡方分组的对应关系附加上去
        stage1_bin_df = pd.DataFrame({'stage1_bin':np.arange(len(stage1_bins_dict))})
        stage1_bin_df['stage2_bin'] = pd.cut(x=stage1_bin_df['stage1_bin'], bins=stage2_bins)
        stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin']
        index = 0
        for i in stage1_bin_df['stage2_bin'].value_counts().sort_index().index:
            stage1_bin_df['stage2_bin_final'] = stage1_bin_df['stage2_bin_final'].replace({i:'Bin_'+"%03d"%(index)})
            index += 1
        for i in stage1_bins_dict.keys():
            stage2_bins_dict[i] = stage1_bin_df['stage2_bin_final'][stage1_bin_df['stage1_bin'] == stage1_bins_dict.get(i)].values[0]
        ##决策树跟卡方同理
        
    ##把special_attribute的单独分箱加进去
    for i in special_attribute:
        stage2_bins_dict[i] = "Bin_special"
        
    return stage2_bins_dict

def category_var_binning_with_bins(data, var, label, bins, woe_shift=0.000001):
    '''
    根据用户给出的分箱节点进行类别型单变量进行分箱操作 并进行woe转换和iv计算
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    bins: 分箱节点 list
    woe_shift: 为防止极端情况出现 在计算woe的时候加上一个小常数
    
    
    return: 
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    IV: 最终的IV值 float
    bins: 分箱节点 list
    '''
    print('正在进行类别型变量{0}的IV计算工作'.format(var))
    ##拷贝一份数据出来 防止篡改原本的数据
    df = data[[var, label]].copy()
    reverse_bins = dict_reverse(bins)
    for i in reverse_bins.keys():
        reverse_bins[i] = ','.join(list(map(str,reverse_bins.get(i))))
    ##计算出对应的箱
    df['bins'] = df[var].replace(bins)
    ##进行实际的分箱工作 并计算odds woe IV等指标
    bins_df = df[['bins',label]].groupby(['bins'], as_index=True).agg([np.sum,len])
    bins_df.columns = ['num_1', 'num_01']
    bins_df['num_0'] = bins_df['num_01'] - bins_df['num_1']
    ##给出每个箱中的包含哪些类别
    bins_df['contains_values'] = pd.Series(bins_df.index).replace(reverse_bins).values
    bins_df = bins_df[['contains_values','num_0', 'num_1', 'num_01']]
    bins_df['pct_0_row'] = bins_df['num_0'] / bins_df['num_01']
    bins_df['pct_1_row'] = bins_df['num_1'] / bins_df['num_01']
    bins_df['pct_0_col'] = bins_df['num_0'] / bins_df['num_0'].sum()
    bins_df['pct_1_col'] = bins_df['num_1'] / bins_df['num_1'].sum()
    
    ##1作为坏用户标签是我们的响应变量 woe越大说明坏用户越多
    bins_df['odds'] = bins_df['pct_1_row'] / bins_df['pct_0_row']
    bins_df['woe'] = np.log(bins_df['pct_1_col'] / bins_df['pct_0_col'])
    bins_df['miv'] = (bins_df['pct_1_col'] - bins_df['pct_0_col']) * bins_df['woe']
    ##由于容易存在小类别样本中只有一种的情况 这里对np.inf做相应的处理 暂时以0考虑
    bins_df['woe'] = bins_df['woe'].replace({np.inf:0,-np.inf:0})
    bins_df['miv'] = bins_df['miv'].replace({np.inf:0,-np.inf:0})
    ##计算IV
    IV = bins_df['miv'].sum()
    bins_df['IV'] = [IV] * bins_df.shape[0]
    ##给出是否单调的判断
    bins_df['is_monotonic'] = [is_monotonic(bins_df['woe'].values)] * bins_df.shape[0]
    ##TODO 给出更好的方向指标
    return bins_df, IV, bins


def category_var_binning(data, var, label, max_interval=10, method='default',special_attribute=[],\
                         tree_params={"criterion":"entropy", "max_leaf_nodes":4, "min_samples_leaf":0.01,"random_state":0},
                         woe_shift=0.000001):
    '''
    根据用户指定的方式对类别型单变量进行分箱操作 并进行woe转换和iv计算
    需要注意的点如果类别稀疏并且使用决策树分箱 很有可能会导致分不任何箱而导致报错 请调节好分箱的参数
    data: 数据源 DataFrame
    var: 待分箱的变量 str
    label: 指导分箱的标签 str
    max_interval: 最大分箱数 在method为"default"时不起作用 int 
    method: "default"(按照降序排序后直接每组单独一个分箱); "ChiMerge"(按照降序排序后进行卡方分箱)；"DecisionTree"(按照降序排序后进行决策树分箱)
    special_attribute：不需要进行分箱的特殊点(单独一个箱)   
    tree_params: 决策树分箱所用决策树的参数
    woe_shift: 为防止极端情况出现 在计算woe的时候加上一个小常数
    
    return: 
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    IV: 最终的IV值 float
    bins: 各类别所属的分箱号 dict
    '''
    
    ##拷贝一份数据出来 防止篡改原本的数据
    df = data[[var, label]].copy()
    ##重新计算分组工作
    bins = category_var_bins_combine(df,var,label,max_interval,method,special_attribute,tree_params)  
    ##根据分组计算woe、iv等值
    bins_df, IV, bins = category_var_binning_with_bins(df,var,label,bins,woe_shift)
    return bins_df, IV, bins


    

def category_var_woe_trans(data, var, bins_df, bins):
    '''
	根据已经完成的分箱操作 对单个类别变量进行相应的woe转化工作
    data: 数据源 DataFrame
    var: woe转化目标变量 str
    bins_df: 分箱明细 包含每个箱的用户数量、WOE、odds等 DataFrame
    bins: 各类别所属的分箱号 dict
    
    return:
    woe_array: 转换完成后每个样本对应的woe值 array
	'''
    df = data[[var]].copy()
    ##进行数据转换的对应dict
    trans_dict = dict(zip(bins_df.index.tolist(), bins_df['woe'].values))
    ##进行分箱操作并转换woe
    df['bins'] = df[var].replace(bins)
    df['woe'] = df['bins'].replace(trans_dict)
    return df['woe'].values


