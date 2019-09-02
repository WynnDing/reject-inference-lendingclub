# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:51:24 2019

@author: whyjust
"""

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn2pmml.decoration import ContinuousDomain
from lightgbm import LGBMClassifier
from pgy_model import pipe_train_test_evaluate
from pgy_evaluation import plot_ks_curve,plot_multi_roc_curve_dict_type,\
                    plot_multi_reject_bad_curve_dict_type,\
                    plot_multi_PR_curve_dict_type,plot_density_curve
from pgy_model import model_result_combine
import pandas as pd
import seaborn as sns
sns.set(style='dark',color_codes=True)
import warnings
warnings.filterwarnings('ignore')

class imbalanceOversampleProcess:
    '''
    numericFeature: 列表,数值型特征列表
    OversampleParamDict：字典{key,value},key限定为['RandomSample','Smote','ADASYN',\
                      'SMOTEENN','SMOTETomek']这几种，value为参数dict
    estimator：需训练的模型，参数要进行初始化
    '''
    def __init__(self,numericFeature,OversampleParamDict,estimator):
        self.numericFeature = numericFeature
        self.OversampleParamDict = OversampleParamDict
        self.estimator = estimator
        self.dataTranformer = DataFrameMapper([(self.numericFeature,\
                        [ContinuousDomain(),SimpleImputer(strategy='mean'), StandardScaler()])])
    
    def _generateModel(self,key,paramDict):
        if key == 'RandomSample':
            self.model = RandomOverSampler(**paramDict)
        elif key == 'Smote':
            self.model = SMOTE(**paramDict)
        elif key == 'ADASYN':
            self.model = ADASYN(**paramDict)
        elif key == 'SMOTEENN':
            self.model = SMOTEENN(**paramDict)
        elif key == 'SMOTETomek':
            self.model = SMOTETomek(**paramDict)
        else:
            assert key not in ['RandomSample','Smote','ADASYN',\
                      'SMOTEENN','SMOTETomek'],'请输入RandomSample,Smote,\
                               ADASYN,SMOTEENN,SMOTETomek中任意一种!'
            
    def _fitSample(self,X,y):
        XTransform = self.dataTranformer.fit_transform(X)
        assert len(self.OversampleParamDict)==1,'只支持单模型输出，字典只能放一组模型参数!'
        for key,value in self.OversampleParamDict.items():
            self._generateModel(key,value) 
        X_train,y_train = self.model.fit_sample(XTransform,y)
        self.X_train_sample = pd.DataFrame(data=X_train,columns=self.numericFeature)
        self.y_train_sample = y_train
    
    def fit(self,X,y):
        self._fitSample(X,y)
        self.estimator.fit(self.X_train_sample,self.y_train_sample)

    def predict_proba(self,X):
        XTransformTest= self.dataTranformer.transform(X)
        X_test = pd.DataFrame(data=XTransformTest,columns=self.numericFeature)
        self.predictResult = self.estimator.predict_proba(X_test)
        return self.predictResult

# 1 读取数据 
train_data = pd.read_csv(r'E:\XM_data\xm_anomaly_train_label.csv')
test_data = pd.read_csv(r'E:\XM_data\xm_anomaly_test_label.csv') 

# 2 拆分训练集
train_data = train_data[train_data.month_status.isin([0,1])]
test_data = test_data[test_data.month_status.isin([0,1])]
drop_feature = ['risk_time','consumer_no','month_status','first_status','data_type']
train_x = train_data.loc[:,~train_data.columns.isin(drop_feature)]
train_y = train_data.loc[:,'month_status']

# 3 拆分测试集
drop_feature = ['risk_time','consumer_no','month_status','first_status','data_type']
test_x = test_data.loc[:,~test_data.columns.isin(drop_feature)]
test_y = test_data.loc[:,'month_status']

# 4 不平衡样本训练
numericFeature = train_x.columns.tolist()
OversampleRandom5 = {'RandomSample':{'ratio':0.5,'random_state':10}}
OversampleRandom4 = {'SMOTEENN':{'ratio':0.4,'random_state':10}}
OversampleRandom3 = {'Smote':{'ratio':0.3,'random_state':10}}


lgb = LGBMClassifier(boosting_type='gbdt',learning_rate=0.1, max_depth=2,n_estimators=500,\
                               n_jobs=-1,objective='binary',importance_type = 'gain',min,\
                               random_state=10)
imbRandom5 = imbalanceOversampleProcess(numericFeature,OversampleRandom5,lgb)
imbRandom4 = imbalanceOversampleProcess(numericFeature,OversampleRandom4,lgb)
imbRandom3 = imbalanceOversampleProcess(numericFeature,OversampleRandom3,lgb)

# 5 评估流程
data_dict = {
        'train':{'X':train_x,'y':train_y},
        'test':{'X':test_x,'y':test_y}
}
model_detail_result = {}
model_statistic_result = {}

model_detail_result5,model_statistic_result5 = pipe_train_test_evaluate(data_dict,imbRandom5)
model_detail_result4,model_statistic_result4 = pipe_train_test_evaluate(data_dict,imbRandom4)
model_detail_result3,model_statistic_result3 = pipe_train_test_evaluate(data_dict,imbRandom3)


model_predict_result = model_result_combine({'RandomSample':model_detail_result5,\
                                             'SMOTEENN':model_detail_result4,\
                                             'Smote':model_detail_result3},'test')
### ks曲线
ks = plot_ks_curve(model_predict_result.get('SMOTEENN').get('predict'),\
                   model_predict_result.get('SMOTEENN').get('true'),n=10,return_graph=True)
## roc曲线
roc_dict,auc_dict = plot_multi_roc_curve_dict_type(model_predict_result)
## 通过率vs拒绝率曲线
bad_rate_result = plot_multi_reject_bad_curve_dict_type(model_predict_result)
## PR曲线
plot_multi_PR_curve_dict_type(model_predict_result)
### 预测概率目的曲线
plot_density_curve(model_predict_result.get('SMOTEENN').get('true'),\
                   model_predict_result.get('SMOTEENN').get('predict'))

         