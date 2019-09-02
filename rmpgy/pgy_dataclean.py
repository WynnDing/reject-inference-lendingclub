# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
import json
from pgy_utils import timecount,check_unique,check_non_intersect
from pgy_preprocess import model_y_generator

@timecount()
def data_clean(config_dict,save_var_type_dict=True):  
    """
    read rawdata lists and then combine them, based on filter condition to get clean data
    1、读取原始数据（若多数据源，那么进行数据聚合），把\\N转换成缺失值,增加一列source_index列；
    2、读取配置文件解析出当前数据集中的y相关、连续型、类别型3种变量
    3、去掉不需要的数据列和数据表
    4、特殊类别变量和异常值处理
    5、Y值相关变量处理
    6、指定变量的缺失值添补
    7、训练集和测试集划分（跨时间）
    8、去掉在某个数据集上缺失百分百的变量
    9、把清洗好的数据集输出到本地目录 然后返回保留的变量信息
    
    config_dict需要包含以下key和value:
    1、data_path数据所在文件的绝对地址
    2、data_list数据文件的名称 只能读csv文件 .csv不用加上
    3、config_name变量配置文件的名称 只能读xlsx文件 .xlsx要加上
    4、train_output_path最终训练数据的输出名称 
    5、test_output_path最终测试数据的输出名称
    6、data_select筛选阐述 包含要删除的列list 要删除的表list 以及要筛选的时间段
    7、source_index数据源列名
    8、time_index时间列列名
    9、sample_index样本id列列名
    
    config_dict现成的例子:
    config_dict = {'data_path':r'E:\rmpgy\data',
               'data_list':['data_a','data_b'],
               'config_name':'var_config_0306.xlsx',
               'train_output_path':'train_clear.csv',
               'test_output_path':'test_clear.csv',
               'var_type_dict_output_path':'var_type_dict.json',
               'data_select':{'table_to_drop':['arc_mxdata_taobao','arc_kexin_xbehavior'],
                              'var_to_drop':['zm_score'],
                              'ext_var':['first_login_phone_type','first_login_time','fitsr_register_time','gmt_mobile','last_login_phone_type','last_login_time','nation'],
                              'time_to_stay':{'data_a_train':['2018-10-01 00:00:00','2018-11-30 23:59:59'],
                                              'data_a_test':['2018-12-01 00:00:00','2018-12-10 23:59:59'],
                                              'data_b_train':['2018-10-01 00:00:00','2018-11-30 23:59:59'],
                                              'data_b_test':['2018-12-01 00:00:00','2018-12-10 23:59:59']
                                              }},
               'source_index':'source_index', ##数据源index的列名
               'time_index':'risk_time', ##时间index的列名
               'sample_index':'consumer_no', ##样本index的列名
               'y_type':'type_2', ##通过哪种方式计算样本标签 model_y_generator函数参数
               'is_grey_good':True, ##通过哪种方式计算样本标签 model_y_generator函数参数
               'y_column':'user_type', ##通过哪种方式计算样本标签 model_y_generator函数参数
               'save_all_data':True ##是否要保留全部数据 如果是False 会把无标签的样本都删除 如果要做拒绝推断 建议保留全部样本然后自己进行筛选
               }
    """
    ##参数
    ##数据所在的地址
    data_path = config_dict.get('data_path')
    
    ##需要读取的数据名称list 只可读取csv文件
    data_list = config_dict.get('data_list')
    ##变量配置文件
    config_name = config_dict.get('config_name')
    ##训练数据输出名称
    train_output_path = config_dict.get('train_output_path')
    ##测试数据输出名称
    test_output_path = config_dict.get('test_output_path')
    ##变量筛选结果输出名称
    var_type_dict_output_path = config_dict.get('var_type_dict_output_path')
    ##筛选参数
    data_select = config_dict.get('data_select')
    #需要保留的列      
    source_index = config_dict.get('source_index')
    time_index = config_dict.get('time_index')
    sample_index = config_dict.get('sample_index')
    id_col_to_retain = [source_index,time_index,sample_index]
    y_type = config_dict.get('y_type','type_1')
    is_grey_good = config_dict.get('is_grey_good',False)
    y_column = config_dict.get('y_column','user_type')
    
    ##数据筛选函数
    def data_time_select(orig_data,time_to_stay,plat_name):
        start_time = dt.strptime(time_to_stay.get(plat_name+'_train')[0],'%Y-%m-%d %H:%M:%S')
        end_time = dt.strptime(time_to_stay.get(plat_name+'_test')[1],'%Y-%m-%d %H:%M:%S')
        orig_data = orig_data[(orig_data['risk_time'] >= start_time)&(orig_data['risk_time'] <= end_time)]
        return orig_data
    
    def train_test_select(orig_data,select_type,time_to_stay):
        output_data = pd.DataFrame()
        for i in np.arange(len(data_list)):
            temp_data = orig_data[orig_data[source_index] == data_list[i]]
            start_time = dt.strptime(time_to_stay.get(data_list[i]+'_'+select_type)[0],'%Y-%m-%d %H:%M:%S')
            end_time = dt.strptime(time_to_stay.get(data_list[i]+'_'+select_type)[1],'%Y-%m-%d %H:%M:%S')
            temp_data = temp_data[(temp_data['risk_time'] >= start_time)&(temp_data['risk_time'] <= end_time)]
            if i == 0:
                output_data = temp_data
            else:
                output_data = pd.concat([output_data,temp_data])
        return output_data
    
    
    ##一、读取原始数据（若多平台数据，进行数据聚合），把\\N转换成缺失值,增加一列plat列；
    print('读取原始数据'.center(50, '='))
    raw_data = pd.DataFrame()
    for i in np.arange(len(data_list)):
        print('正在读取'+data_list[i])
        path  = os.path.join(data_path,data_list[i]+'.csv')
        temp_data = pd.read_csv(path,header = 0, encoding = 'utf-8',na_values = '\\N')
        temp_data[source_index] = data_list[i]
        temp_data['risk_time'] = temp_data['risk_time'].astype('datetime64[ns]')
        temp_data = data_time_select(temp_data,data_select.get('time_to_stay',[]),data_list[i])
        print('筛选结果区间:' + dt.strftime(temp_data['risk_time'].min(),'%Y-%m-%d') + '~' + dt.strftime(temp_data['risk_time'].max(),'%Y-%m-%d'))
        if i == 0:
            raw_data = temp_data
        else:
            raw_data = pd.concat([raw_data,temp_data])
            
    raw_data.reset_index(drop=True,inplace=True)
    ##二、读取配置文件
    print('读取配置文件'.center(50, '='),'\n')
    ##读取配置文件
    var_config = pd.read_excel(os.path.join(data_path,config_name),sheet_name=u'全量变量分类细节')
    ##解析出当前数据集中的y相关、连续型、类别型3种变量
    y_label_var = [i for i in list(raw_data.columns) if i in list(var_config['var_en'][(var_config['is_useful'] == 1)&(var_config['is_y_related'] == 1)])]
    numeric_var = [i for i in list(raw_data.columns) if i in list(var_config['var_en'][(var_config['is_useful'] == 1)&(var_config['is_continuous'] == 1)])]
    category_var = [i for i in list(raw_data.columns) if i in list(var_config['var_en'][(var_config['is_useful'] == 1)&(var_config['is_categorical'] == 1)])]
    ext_var = [i for i in list(raw_data.columns) if i in data_select.get('ext_var',[])]
    print('ext_var在数据集中找到如下几个:%s'%ext_var)
    ##特殊类别转换文件
    special_replace = pd.read_excel(os.path.join(data_path,config_name),sheet_name=u'字符转换规则')
    special_replace = special_replace[(special_replace['label_cn'] == '-')&(special_replace['var_en'].isin(numeric_var))]
       
    ##（2）去掉不需要的数据列
    print('去掉不需要的数据列'.center(50, '='))
    for col in data_select.get('var_to_drop',[]):
        print('正在删除数据列'+col)
        numeric_var = [i for i in numeric_var if i not in [col]]
        category_var = [i for i in category_var if i not in [col]]
        
    ##（3）去掉不需要的数据表
    print('去掉不需要的数据表'.center(50, '='))
    for table in data_select.get('table_to_drop',[]):
        print('正在删除数据表'+table)
        table_col = var_config['var_en'][var_config['table_en'] == table].values
        numeric_var = [i for i in numeric_var if i not in table_col]
        category_var = [i for i in category_var if i not in table_col]
        
    ##（4）按需求转换Y
    ## 如果y_label_var小于等于1个 那么可以认为数据里面没有足够转换标签的数据 为拒绝样本 标记为-3
    if len(y_label_var) > 1:
        raw_data[y_column] = model_y_generator(raw_data[y_label_var],y_type,is_grey_good)
        y_label_var = [y_column]
    else:
        raw_data[y_column] = -3
        y_label_var = [y_column]
        
    if config_dict['save_all_data']:
        print('保留无标签数据'.center(50, '='))
        raw_data = raw_data
    else:
        raw_data = raw_data[raw_data[y_column].isin([0,1])]
    
    ##三、把代表缺失值和异常值的数据转成缺失值，同时修正变量的类型；
    ##对特殊类别的变量，进行优先特殊转换
    print('特殊类别变量转换'.center(50, '='),'\n')
    for var in special_replace['var_en']:
        if var in numeric_var + category_var:
            print(('正在转换特殊字段'+var+'的异常值').center(50, ' '))
            raw_data[var] = raw_data[var].replace({special_replace['label_cn'][special_replace['var_en']==var].values[0]:special_replace['label_num'][special_replace['var_en']==var].values[0]})
    if 'tmall_level' in numeric_var + category_var:
        print(('正在转换特殊字段'+'tmall_level'+'的异常值').center(50, ' '))
        raw_data['tmall_level'] = raw_data['tmall_level'].replace({'-':np.nan})
        raw_data['tmall_level'] = raw_data['tmall_level'].astype('float64')
        raw_data['tmall_level'][raw_data['tmall_level'] <= 0] = np.nan
    
    ##对数值型 小于等于anomaly_value记为异常值 转换缺失值  并统一成float类型
    print('数值型异常处理'.center(50, '='))
    for var in numeric_var:
        ##进行异常值转换
        if ~var_config['anomaly_value'].isna()[var_config['var_en'] == var].values[0]: ##判断是否有标记的异常值
            anomaly_value = var_config['anomaly_value'][var_config['var_en'] == var].values[0]
            print(('正在转换连续字段'+var+'的异常值:'+str(anomaly_value)+' 类型:'+str(type(anomaly_value)).split("'")[1]))
            if type(anomaly_value) is str:
                raw_data[var] = raw_data[var].replace(to_replace=anomaly_value,value=np.nan)
            else:
                raw_data[var][raw_data[var] <= anomaly_value] = np.nan
        ##统一numeric_var的类型 
        if raw_data[var].dtype != np.float64:
            print('正在转换连续字段'+var+'的字段类型')
            raw_data[var] = raw_data[var].astype(np.float64)
    ##对类别型 等于anomaly_value记为异常值 转换成缺失值
    print('类别型异常处理'.center(50, '='))
    for var in category_var:
        if ~var_config['anomaly_value'].isna()[var_config['var_en'] == var].values[0]: ##判断是否有标记的异常值
            anomaly_value = var_config['anomaly_value'][var_config['var_en'] == var].values[0]
            print(('正在转换类别字段'+var+ '的异常值:'+str(anomaly_value)+' 类型:'+str(type(anomaly_value)).split("'")[1]))
            if type(anomaly_value) is str:
                raw_data[var] = raw_data[var].replace(to_replace=anomaly_value,value=np.nan)
    
    # 常数类别变量的去除
    # category_var_new = [cate for cate in list(category_var) if len(raw_data[cate].value_counts().index) != 1] 
    
    
    ##五、缺失值的填补
    ##进行部分指定变量的缺失值填补工作
    print('缺失值填补'.center(50, '='),'\n')
    trans_var = var_config['var_en'][~var_config['na_fill'].isna()].values
    for var in trans_var:
        raw_data[var] = raw_data[var].replace(to_replace=np.nan,value=var_config['na_fill'][var_config['var_en']==var].values[0])
    
    ##六、训练集验证集划分
    print('训练集验证集划分'.center(50, '='))
    train_data = train_test_select(raw_data,'train',data_select.get('time_to_stay',[]))
    test_data = train_test_select(raw_data,'test',data_select.get('time_to_stay',[]))
    print('训练集大小:' + str(len(train_data)) + ' 验证集大小:' + str(len(test_data)))
    
    
    
    ##去掉缺失值百分之百的变量
    print('删除在某个数据集上缺失为百分百的变量'.center(50, '='))
    train_na_percent = train_data.apply(lambda x:np.sum(x.isna())/len(x))
    test_na_percent = test_data.apply(lambda x:np.sum(x.isna())/len(x))
    na_drop_var =  [i for i in set(train_na_percent.index[train_na_percent==1].values) | set(test_na_percent.index[test_na_percent==1].values) if i in numeric_var+category_var]
    print('删除数值变量:%s'%[i for i in numeric_var if i in na_drop_var])
    print('删除类别变量:%s'%[i for i in category_var if i in na_drop_var] )
    print('删除ext变量:%s'% [i for i in ext_var if i in na_drop_var]   )
    numeric_var = [i for i in numeric_var if i not in na_drop_var]
    category_var = [i for i in category_var if i not in na_drop_var]    
    ext_var = [i for i in ext_var if i not in na_drop_var]           
     
    ##检查不同组内的列名是否有重复
    if not check_unique(numeric_var):
        print('numeric_var存在重复列')
    if not check_unique(category_var):
        print('category_var存在重复列')
    if not check_unique(ext_var):
        print('ext_varr存在重复列')
    ##检查不同组间是否列名有重复
    check_non_intersect(numeric_var,category_var)
    check_non_intersect(ext_var,category_var)
    check_non_intersect(ext_var,numeric_var)
    
    ##七、输出结果
    print('输出最终结果'.center(50, '='))
    print('数值型、类别型、ext变量:',len(numeric_var),len(category_var),len(ext_var),'总数据集维度：',raw_data[numeric_var+category_var+ext_var+y_label_var].shape)        
    print('训练集输出'.center(50, '-'))
    for plat in data_list:
        print(plat+'时间区间:'+dt.strftime(train_data[train_data[source_index] == plat]['risk_time'].min(),'%Y-%m-%d')+'~'+dt.strftime(train_data[train_data[source_index] == plat]['risk_time'].max(),'%Y-%m-%d')+\
              ' 数据集大小:'+str(len(train_data[train_data[source_index] == plat])))
    print('验证集输出'.center(50, '-'))
    for plat in data_list:
        print(plat+'时间区间:'+dt.strftime(test_data[test_data[source_index] == plat]['risk_time'].min(),'%Y-%m-%d')+'~'+dt.strftime(test_data[test_data[source_index] == plat]['risk_time'].max(),'%Y-%m-%d')+\
              ' 数据集大小:'+str(len(test_data[test_data[source_index] == plat])))    
    
    train_data[id_col_to_retain+numeric_var+category_var+ext_var+y_label_var].to_csv(os.path.join(data_path,train_output_path),index=False)
    test_data[id_col_to_retain+numeric_var+category_var+ext_var+y_label_var].to_csv(os.path.join(data_path,test_output_path),index=False)
    var_type_dict = {'source_index':source_index,
                     'time_index':time_index,
                     'sample_index':sample_index,
                     'numeric_var':numeric_var,
                     'category_var':category_var,
                     'ext_var':ext_var,
                     'y_label_var':y_label_var[0]
                     }
    if save_var_type_dict:
        with open(os.path.join(data_path,var_type_dict_output_path),'w') as f:
            f.write(json.dumps(var_type_dict))