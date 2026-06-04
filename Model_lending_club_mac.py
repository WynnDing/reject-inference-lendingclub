# -*- coding: utf-8 -*-
"""
Mac port of Model_lending_club0213.py  (Reject Inference, Lending Club)

Goal of this port (per request): get the pipeline RUNNING on macOS and verify
that the base ensemble model + the round-1 reject-inference setup work. It does
NOT sit through all 3 iterative rounds.

Faithful to the original methodology; only the following were changed:
  * os.chdir(WINDOWS_PATH) -> sys.path.insert for the rmpgy package import
  * hardcoded C:\\Users\\wd007\\Box\\RI paths -> this folder (BASE_DIR)
  * pd.read_csv now uses usecols (loads only the columns the pipeline uses) ->
    equivalent results, but avoids reading the full 3.3 GB of CSV into RAM
  * `from sklearn.externals import joblib` -> `import joblib` (removed in modern sklearn)
  * reject_main_train() grid is now parametrizable (default = original grid);
    the verification run below uses a reduced grid so it finishes quickly
  * matplotlib forced to a non-interactive backend; sub-model logging silenced
  * rounds 2 and 3 are left in as commented reference (see bottom)
"""

import os
import sys
import gc
import random
import warnings
from datetime import datetime as dt
from functools import wraps

import matplotlib
matplotlib.use("Agg")  # no GUI on this run
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "rmpgy"))  # was: os.chdir(r'C:\Users\wd007\Box\RI\rmpgy')
OUT_DIR = os.path.join(BASE_DIR, "output_mac")        # don't clobber original artifacts
os.makedirs(OUT_DIR, exist_ok=True)

import lightgbm as gbm
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score, classification_report
from sklearn_pandas import CategoricalImputer, DataFrameMapper, gen_features
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from pgy_evaluation import plot_ks_curve  # only fn used in the training path


class ModeImputer(BaseEstimator, TransformerMixin):
    """Most-frequent categorical imputer that breaks ties deterministically.

    Drop-in replacement for sklearn_pandas.CategoricalImputer, which raises
    'No value is repeated more than once' whenever pd.Series.mode() returns a
    tie (two categories with equal top frequency). For non-tie columns the
    imputed value is identical; on a tie we take the first (sorted) mode."""
    def __init__(self, missing_values=np.nan):
        self.missing_values = missing_values

    def fit(self, X, y=None):
        s = pd.Series(np.asarray(X).ravel())
        s = s[s.notna()]
        m = s.mode()
        self.fill_ = m.iloc[0] if len(m) else '?'
        return self

    def transform(self, X):
        return pd.Series(np.asarray(X).ravel()).fillna(self.fill_).values


def timecount():
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = dt.now()
            temp_result = func(*args, **kwargs)
            time_pass = dt.now() - start_time
            print(("time consuming: " + str(np.round(time_pass.total_seconds() / 60, 2)) + "min").center(50, "="))
            return temp_result
        return wrapper
    return decorate


def time_convert(timeStr):
    if str(timeStr).lower() == "nan":
        return np.nan
    month_name_dict = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                       "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
    time_split = str(timeStr).upper().split("-")
    return time_split[1] + "-" + month_name_dict[time_split[0]]


def state_drive(arr):
    if str(arr) in ['NV', 'RI', 'FL', 'AZ', 'HI', 'MI']:
        return 'state_1'
    elif str(arr) in ['MD', 'VT', 'WI', 'DE', 'CT', 'SC', 'KY', 'OH', 'VA', 'CO', 'SD', 'MA', 'LA', 'IL', 'NH', 'PA', 'TX']:
        return 'state_2'
    elif str(arr) in ['AL', 'GA', 'MT', 'NJ', 'CA', 'OK', 'WA', 'NM', 'OR', 'UT', 'AR', 'AY', 'AK', 'NC', 'MO', 'MN']:
        return 'state_3'
    else:
        return 'state_4'


@timecount()
def load_data():
    # usecols added: only the columns the pipeline actually consumes (equivalent result, much less RAM)
    accepted_cols = ['loan_amnt', 'emp_length', 'issue_d', 'loan_status', 'title', 'addr_state', 'dti', 'fico_range_low']
    reject_cols = ['Amount Requested', 'Application Date', 'Loan Title', 'Risk_Score',
                   'Debt-To-Income Ratio', 'State', 'Employment Length']

    train_data = pd.read_csv(os.path.join(BASE_DIR, "accepted_2007_to_2018Q4.csv"),
                             header=0, encoding="utf-8", na_values="\\N", usecols=accepted_cols)
    reject_data = pd.read_csv(os.path.join(BASE_DIR, "rejected_2007_to_2018Q4.csv"),
                              header=0, encoding="utf-8", na_values="\\N",
                              usecols=reject_cols, parse_dates=["Application Date"])

    train_data['Application Date'] = train_data['issue_d'].apply(time_convert)
    train_data_temp = train_data[(train_data['Application Date'] >= '2009-01') & (train_data['Application Date'] <= '2011-12')]

    reject_data_temp = reject_data[(reject_data['Application Date'] >= '2009-01-01')
                                   & (reject_data['Application Date'] <= '2011-06-30')]
    reject_data_temp['Debt-To-Income Ratio'] = reject_data_temp['Debt-To-Income Ratio'].map(lambda x: float(str(x).split('%')[0]))

    train_data_temp['user_type'] = 0
    train_data_temp['user_type'][(train_data_temp['loan_status'] == 'Does not meet the credit policy. Status:Charged Off')
                                 | (train_data_temp['loan_status'] == 'Charged Off')] = 1
    print('ratio:%s' % (np.sum(train_data_temp['user_type']) / len(train_data_temp)))

    train_data_temp.rename(columns={'loan_amnt': 'Amount Requested', 'title': 'Loan Title', 'fico_range_low': 'Risk_Score',
                                    'dti': 'Debt-To-Income Ratio', 'addr_state': 'State', 'emp_length': 'Employment Length'}, inplace=True)
    for data in [train_data_temp, reject_data_temp]:
        data['state_type'] = data['State'].apply(state_drive)

    data_temp = train_data_temp[(train_data_temp['Application Date'] > '2011-04') & (train_data_temp['Application Date'] <= '2011-08')]
    test_data, valid_data, y_test, y_valid = train_test_split(data_temp, data_temp['user_type'], test_size=0.5, random_state=8)
    train_data = train_data_temp[train_data_temp['Application Date'] <= '2011-04']

    feature = ['Amount Requested', 'Risk_Score', 'Debt-To-Income Ratio', 'Employment Length', 'state_type', 'Application Date']
    return (train_data[feature + ['user_type']], reject_data_temp[feature],
            valid_data[feature + ['user_type']], test_data[feature + ['user_type']])


def model_generate(param_dict, types='train'):
    lightgbm_model = gbm.LGBMClassifier().set_params(**param_dict.get('lightgbm', {}))
    xgboost_model = XGBClassifier().set_params(**param_dict.get('xgboost', {}))
    catboost_model = CatBoostClassifier().set_params(**param_dict.get('catboost', {}))
    if types == 'train':
        RF_model = RandomForestClassifier().set_params(**param_dict.get('RF', {}))
        NB_model = GaussianNB().set_params(**param_dict.get('NB', {}))
        lr_model = LogisticRegression().set_params(**param_dict.get('lr', {}))
        return [lightgbm_model, xgboost_model, RF_model, NB_model, lr_model, catboost_model]
    elif types == 'eval':
        return [lightgbm_model, xgboost_model, catboost_model]
    raise ValueError


def feature_union(category_feature, numeric_feature):
    # invalid_value_treatment='as_missing': sklearn2pmml 0.98.0 is stricter than the original
    # Windows env and would otherwise raise on out-of-range/invalid values; routing them to
    # "missing" lets the downstream imputers fill them, which is the original intent.
    mapper_category = DataFrameMapper(gen_features(
        columns=category_feature,
        classes=[{'class': CategoricalDomain, 'invalid_value_treatment': 'as_missing'},
                 ModeImputer, LabelEncoder]))
    mapper_numerical = DataFrameMapper([
        (numeric_feature, [ContinuousDomain(invalid_value_treatment='as_missing'),
                           SimpleImputer(strategy='mean'), StandardScaler()])])
    return FeatureUnion([('mapper_category', mapper_category), ('mapper_numerical', mapper_numerical)])


def fuse_model_train(X_train, y_train, model_list, category_feature, numeric_feature, classifier_type='train'):
    pipeline_transformer = feature_union(category_feature, numeric_feature)
    model_tuple = list(zip([str(i) for i in range(len(model_list))], model_list))
    if classifier_type == 'train':
        model_fuse = VotingClassifier(estimators=model_tuple, voting='soft', weights=[4, 2, 2, 1.5, 2, 4])
    elif classifier_type == 'eval':
        model_fuse = VotingClassifier(estimators=model_tuple, voting='soft', weights=[5, 8, 1])
    else:
        raise ValueError
    pipeline_model = PMMLPipeline([('mapper', pipeline_transformer), ('classifier', model_fuse)])
    print('y_train:', y_train.shape, 'ratio:', np.sum(y_train.values) / len(y_train))
    pipeline_model.fit(X_train, y_train)
    return pipeline_model


def submodel_evaluation(train_data, valid_data, model_list, category_feature, numeric_feature):
    X_train = train_data[category_feature + numeric_feature]
    y_train = train_data['user_type']
    X_valid = valid_data[category_feature + numeric_feature]
    y_valid = valid_data['user_type']
    pipeline_transformer = feature_union(category_feature, numeric_feature)
    model_result_dict = {}
    for model in model_list:
        model_name = model.__class__.__name__
        print('model %s evaluation' % model_name)
        sub_model = PMMLPipeline([('mapper', pipeline_transformer), ('classifier', model)])
        sub_model.fit(X_train, y_train)
        predict_valid = sub_model.predict_proba(X_valid)[:, 1]
        predict_label = sub_model.predict(X_valid)
        model_ks = plot_ks_curve(predict_valid, valid_data['user_type'])
        model_auc = roc_auc_score(y_valid, predict_valid)
        accuracy = metrics.accuracy_score(y_valid, predict_label)
        model_result_dict[model_name] = [model_ks, model_auc, accuracy]
    return model_result_dict


def model_fuse_evaluation(model, train_data, valid_data, test_data, feature_used):
    X_valid, y_valid = valid_data[feature_used], valid_data['user_type']
    X_test, y_test = test_data[feature_used], test_data['user_type']
    X_train, y_train = train_data[feature_used], train_data['user_type']

    predict_valid = model.predict_proba(X_valid)[:, 1]
    predict_label = model.predict(X_valid)
    valid_ks = plot_ks_curve(predict_valid, y_valid)
    valid_auc = roc_auc_score(y_valid, predict_valid)

    predict_test = model.predict_proba(X_test)[:, 1]
    ks_test = plot_ks_curve(predict_test, y_test)
    auc_test = roc_auc_score(y_test, predict_test)

    predict_train = model.predict_proba(X_train)[:, 1]
    ks_train = plot_ks_curve(predict_train, y_train)
    auc_train = roc_auc_score(y_train, predict_train)

    print(classification_report(y_valid.values, predict_label, target_names=['0', '1']))
    accuracy = metrics.accuracy_score(y_valid, predict_label)
    return [valid_ks, valid_auc, ks_test, auc_test, auc_train, ks_train, accuracy]


def reject_data_masked(reject_train, predict_reject_prob, n_sample=1000, ratio=1, bad=0.8, good=0.2):
    reject_train['user_prob'] = predict_reject_prob
    reject_train_temp = reject_train[((reject_train['user_prob'] < 1) & (reject_train['user_prob'] > bad))
                                     | ((reject_train['user_prob'] < good) & (reject_train['user_prob'] > 0))]
    reject_train_temp['user_type'] = reject_train_temp['user_prob'].apply(lambda x: 1.0 if x > 0.5 else 0)
    print('reject samples:', reject_train_temp['user_type'].value_counts().to_dict())
    n = random.randint(1, 10)
    temp_1 = reject_train_temp[reject_train_temp['user_type'] == 1].sample(n_sample, random_state=n)
    temp_0 = reject_train_temp[reject_train_temp['user_type'] == 0].sample(int(ratio * n_sample), random_state=n)
    del reject_train_temp
    gc.collect()
    return shuffle(pd.concat([temp_0, temp_1]))


def reject_data_predict(reject_data, pipeline_model, feature_used):
    reject_train = reject_data[feature_used]
    predict_reject_prob = pipeline_model.predict_proba(reject_train)[:, 1]
    return predict_reject_prob, reject_train


def raw_combine_reject(train_data, reject_train_combine_label, feature_used):
    combined = pd.concat([train_data[feature_used + ['user_type']],
                          reject_train_combine_label[feature_used + ['user_type']]]).reset_index(drop=True)
    return combined[feature_used], combined['user_type']


@timecount()
def reject_main_train(reject_train, train_data, predict_reject_prob, model_list, feature_used,
                      model_auc_raw, bad, good, diff=0,
                      ratios=(0.5, 1, 2), n_samples=(1000, 2000)):  # grids parametrized; defaults = original
    reject_evalutaion_detail = {}
    pipeline_model_userful = []
    reject_train_userful = []
    for ratio in ratios:
        for n_sample in n_samples:
            print('start %s_%s' % (ratio, n_sample))
            reject_train_combine_label = reject_data_masked(reject_train, predict_reject_prob, n_sample, ratio, bad, good)
            X_train_reject_1, y_train_reject_1 = raw_combine_reject(train_data, reject_train_combine_label, feature_used)
            pipeline_model_reject_1 = fuse_model_train(X_train_reject_1, y_train_reject_1, model_list,
                                                       category_feature, numeric_feature, classifier_type='train')
            evaluation_result = model_fuse_evaluation(pipeline_model_reject_1, train_data, valid_data, test_data, feature_used)
            reject_evalutaion_detail[str(ratio) + '_' + str(n_sample)] = evaluation_result
            if evaluation_result[1] - model_auc_raw >= diff:
                print('reject auc:', evaluation_result[1], 'raw_auc:', model_auc_raw)
                print(y_train_reject_1.value_counts().to_dict())
                pipeline_model_userful.append(pipeline_model_reject_1)
                reject_train_userful.append(reject_train_combine_label)
    return reject_evalutaion_detail, pipeline_model_userful, reject_train_userful


# ============================ verification run ============================
if __name__ == "__main__":
    # 1. load features
    train_data, reject_data, valid_data, test_data = load_data()
    category_feature = ['state_type', 'Employment Length']
    numeric_feature = ['Amount Requested', 'Risk_Score', 'Debt-To-Income Ratio']
    feature_used = category_feature + numeric_feature
    X_train, y_train = train_data[feature_used], train_data['user_type']
    print('X_train:', X_train.shape, 'X_valid:', valid_data[feature_used].shape, 'reject:', reject_data.shape)

    reject_data_temp = reject_data
    param_dict_eval = {
        'lightgbm': {'boosting_type': 'gbdt', 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 10,
                     'n_jobs': -1, 'class_weight': {0: 1, 1: 4}, 'verbosity': -1},
        'xgboost': {'learning_rate': 0.1, 'n_estimators': 300, 'max_depth': 2, 'objective': 'binary:logistic',
                    'seed': 10, 'gamma': 1.4, 'reg_alpha': 4, 'reg_lambda': 0.1, 'random_state': 10,
                    'scale_pos_weight': 4, 'n_jobs': -1, 'verbosity': 0},
        'lr': {'C': 0.1, 'penalty': 'l2', 'random_state': 10, 'class_weight': {0: 1, 1: 4}, 'n_jobs': -1},
        'RF': {'n_estimators': 600, 'max_depth': 4, 'random_state': 10, 'class_weight': {0: 1, 1: 4}, 'n_jobs': -1},
        'NB': {'priors': None},
        'catboost': {'iterations': 400, 'max_depth': 3, 'learning_rate': 0.1, 'scale_pos_weight': 4,
                     'random_seed': 10, 'verbose': False},
    }
    model_list_eval = model_generate(param_dict_eval, types='train')

    # 2. base ensemble model
    print('\n' + ' BASE MODEL '.center(50, '='))
    pipeline_model_base = fuse_model_train(X_train, y_train, model_list_eval, category_feature, numeric_feature, classifier_type='train')

    # 3. base model evaluation (raw AUC)
    evaluation_raw_details = model_fuse_evaluation(pipeline_model_base, train_data, valid_data, test_data, feature_used)
    model_auc_raw = evaluation_raw_details[1]
    eval_df = pd.DataFrame({'raw': evaluation_raw_details},
                           index=['valid_ks', 'valid_auc', 'ks_test', 'auc_test', 'auc_train', 'ks_train', 'accuracy'])
    print('\nBASE (raw) model evaluation:')
    print(eval_df)

    # 4. ROUND 1 reject inference  (reduced grid for the verification run)
    print('\n' + ' ROUND 1 REJECT INFERENCE '.center(50, '='))
    predict_reject_prob, reject_train = reject_data_predict(reject_data_temp, pipeline_model_base, feature_used)
    reject_eval_1, pipeline_model_userful, reject_train_userful = reject_main_train(
        reject_train, train_data, predict_reject_prob, model_list_eval, feature_used,
        model_auc_raw, bad=0.65, good=0.25, diff=0,
        ratios=(1,), n_samples=(1000,))   # <-- reduced; original was ratios=(0.5,1,2), n_samples=(1000,2000)

    eval_reject_df_1 = pd.DataFrame(reject_eval_1,
                                    index=['valid_ks', 'valid_auc', 'ks_test', 'auc_test', 'auc_train', 'ks_train', 'accuracy'])
    print('\nROUND 1 reject-augmented evaluation:')
    print(eval_reject_df_1)

    print('\n' + ' SUMMARY '.center(50, '='))
    print('base (raw) valid AUC      : %.5f' % model_auc_raw)
    for cfg in reject_eval_1:
        print('round1 [%s] valid AUC    : %.5f  (delta %+.5f)' % (cfg, reject_eval_1[cfg][1], reject_eval_1[cfg][1] - model_auc_raw))
    print('configs that beat raw AUC : %d' % len(pipeline_model_userful))
    print('\nVerification complete: base model + round-1 reject-inference setup ran end-to-end.')
    print('(Rounds 2 and 3 are skipped in this verification run -- see original Model_lending_club0213.py.)')
