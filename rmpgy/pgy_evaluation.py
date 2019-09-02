# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc ,precision_recall_curve,average_precision_score,roc_auc_score
from sklearn.model_selection import learning_curve,validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns


'''
    本代码块中大部分只适用于二分类信用模型评估 模型要求能够输出概率 0代表好 1代表坏
'''


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    用于绘制混淆矩阵图
    y_true: 真实预测结果 Series或者array
    y_pred: 模型预测结果（非概率）Series或者array
    classes: 预测类别的名称 Series或者array
    normalize: 是否进行标准化 boolean
    title： 图形标题 str
    you can see example on https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()


##  画KS图和输出结果的ks值
def plot_ks_curve(preds, labels, is_score=False, n=100, return_value=True,return_graph=False,return_table=False):
    """
    Plot KS and return ks result
    :param preds: predict probability or scores
    :type preds: list
    :param labels: the actual labels, 1 is bad, 0 is good
    :type labels: list
    :param is_score: False by default indicating that when
    the preds value is probability. Otherwise, preds is score.
    :type is_score: bool
    :param n:
    :type n:
    :param return_value return_graph  return_table is bool
    :rtype: bool
    """

    ksds = pd.DataFrame({'bad': labels, 'pred': preds})
    ksds['good'] = 1 - ksds.bad

    if is_score:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    else:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if is_score:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    else:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds,
                        columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    if return_graph:
        plt.figure(figsize=(8,6))
        print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(
        np.round(ks_pop, 4)))
        # chart
        plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
                 color='blue', linestyle='-', linewidth=2)
        plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
                 color='red', linestyle='-', linewidth=2)
        plt.plot(ksds.tile, ksds.ks, label='ks',
                 color='green', linestyle='-', linewidth=2)
        plt.axvline(ks_pop, color='gray', linestyle='--')
        plt.axhline(ks_value, color='green', linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue',
                    linestyle='--')
        plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red',
                    linestyle='--')
        plt.title('KS=%s ' % np.round(ks_value, 4) +
                  'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)
        plt.show()
    if return_value:
        return ks_value
    elif return_table:
        return ksds,ks_value
    else:
        assert (return_table or return_graph or return_value),'请设置参数为True'
        return None

# 绘制ROC曲线
def plot_roc_curve(y_true,y_predict,return_value=True,return_graph=False):
    '''
    用户绘制ROC曲线图 并返回auc值
    y_true: 真实标签 非概率 Series或者array
    y_predict: 模型预测的结果 概率 Series或者array
    return_value: 是否需要返回值
    return_graph：是否需要画图
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_predict)  ###计算真正率和假正率  
    roc_auc = auc(fpr, tpr) ###计算auc的值  
    if return_graph:
        plt.figure()
        plt.figure(figsize=(5, 5))
        lw=2
        plt.plot(fpr, tpr, color='darkorange', lw =lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Roc Curve')
        plt.legend(loc="lower right")
        plt.show()
    if return_value:
        return roc_auc
    
# 多模型ROC绘制 
def plot_multi_roc_curve(model_dict,X,y):
    """
    用于多个模型绘制ROC曲线 并返回每个模型的AUC结果 即将废弃！！！
    model_dict: 'model_name':model组成的dict
    X: X数据
    y: Y数据
    
    return:
    probability_result: 模型预测的结果和真实标签 pd.DataFrame
    roc_dict: 每个模型的roc曲线详细内容 dict 例如{'model_name':{'fpr':fpr,'tpr':tpr,'thresholds':thresholds}} 
    auc_dict: 每个模型的auc dict 例如{'model_name':auc_value}
    """
    plt.figure(figsize=(8,6))
    probability_result = pd.DataFrame()
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        y_predict_prob = model.predict_proba(X)[:,1]
        probability_result[model_name] = y_predict_prob
    probability_result['y_label'] = y.values
    roc_dict = {}
    auc_dict = {}
    for column in probability_result.columns[:-1]:
        y_predict_prob = probability_result[column]       
        fpr,tpr,thresholds = roc_curve(y,y_predict_prob)
        auc = roc_auc_score(y, y_predict_prob)
        roc_dict[model_name] = {'fpr':fpr,'tpr':tpr,'thresholds':thresholds}
        auc_dict[model_name] = auc
        plt.plot(roc_dict[model_name]['fpr'],roc_dict[model_name]['tpr'],label='{} auc:{:.4f}'\
                 .format(model_name,auc_dict[model_name]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC Curves for Classifiers')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 4)
    plt.show()
    return probability_result,roc_dict,auc_dict    

# 多模型ROC绘制 dict版本
def plot_multi_roc_curve_dict_type(predict_dict):
    """
    用于多个模型绘制ROC曲线 并返回每个模型的AUC结果
    predict_dict 多个模型的预测结果和真实结果组成的dict 形如{'model1':{'predict':[],'true':[]},.....}
    
    return:
    roc_dict: 每个模型的roc曲线详细内容 dict 例如{'model_name':{'fpr':fpr,'tpr':tpr,'thresholds':thresholds}} 
    auc_dict: 每个模型的auc dict 例如{'model_name':auc_value}
    """
    plt.figure(figsize=(8,6))
    roc_dict = {}
    auc_dict = {}
    for model_name in predict_dict.keys():
        y_predict_prob = predict_dict.get(model_name).get('predict')
        y_true = predict_dict.get(model_name).get('true')   
        fpr,tpr,thresholds = roc_curve(y_true,y_predict_prob)
        auc = roc_auc_score(y_true, y_predict_prob)
        roc_dict[model_name] = {'fpr':fpr,'tpr':tpr,'thresholds':thresholds}
        auc_dict[model_name] = auc
        plt.plot(roc_dict[model_name]['fpr'],roc_dict[model_name]['tpr'],label='{} auc:{:.4f}'\
                 .format(model_name,auc_dict[model_name]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC Curves for Classifiers')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 4)
    plt.show()
    return roc_dict,auc_dict



def plot_multi_reject_bad_curve(model_dict,X,y):
    """
    用于多个模型绘制拒绝率vs坏样本率曲线 并返回每个模型的详细结果
    model_dict: 'model_name':model组成的dict
    X: X数据
    y: Y数据
    
    return:
    probability_result: 模型预测的结果和真实标签 pd.DataFrame
    bad_rate_result: 不同模型在给定通过率下的坏用户比率 pd.DataFrame
    """
    plt.figure(figsize=(8,6))
    probability_result = pd.DataFrame()
    bad_rate_result = pd.DataFrame()
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        y_predict_prob = model.predict_proba(X)[:,1]
        probability_result[model_name] = y_predict_prob
    probability_result['y_label'] = y.values
    for column in probability_result.columns[:-1]:
        model_probability = probability_result[[column,'y_label']].sort_values(column)
        bad_rate_list = []
        for x in np.arange(0.01,1,0.01):
            threshold = model_probability[column].quantile(x)
            user_through = model_probability[model_probability[column]<threshold]
            bad_rate = user_through['y_label'][user_through['y_label']==1].count()/user_through['y_label'].count()
            bad_rate_list.append(bad_rate)
        bad_rate_result[column] = bad_rate_list
        plt.plot(np.arange(0.01,1,0.01).tolist(),bad_rate_result[column],label='{}_bad_rate'.format(column))
    plt.gca().invert_xaxis()
    plt.title('Model Reject Rate Compare')
    plt.xlabel('Pass Rate')
    plt.ylabel('Model Bad Rate')
    plt.legend(loc = 1)
    
    bad_rate_result.index = np.arange(0.01,1,0.01)
    return probability_result,bad_rate_result

def plot_multi_reject_bad_curve_dict_type(predict_dict):
    """
    用于多个模型绘制拒绝率vs坏样本率曲线 并返回每个模型的详细结果
    predict_dict 多个模型的预测结果和真实结果组成的dict 形如{'model1':{'predict':[],'true':[]},.....}
    
    return:
    bad_rate_result: 不同模型在给定通过率下的坏用户比率 pd.DataFrame
    """
    plt.figure(figsize=(8,6))
    bad_rate_result = pd.DataFrame()
    for model_name in predict_dict.keys():
        y_predict_prob = predict_dict.get(model_name).get('predict')
        y_true = predict_dict.get(model_name).get('true')
        bad_rate_list = []
        for x in np.arange(0.01,1,0.01):
            threshold = pd.Series(y_predict_prob).quantile(x)
            user_through = y_true[y_predict_prob<threshold]
            bad_rate = np.sum(user_through)/len(user_through)
            bad_rate_list.append(bad_rate)
        bad_rate_result[model_name] = bad_rate_list
        plt.plot(np.arange(0.01,1,0.01).tolist(),bad_rate_result[model_name],label='{}_bad_rate'.format(model_name))
    plt.gca().invert_xaxis()
    plt.title('Model Reject Rate Compare')
    plt.xlabel('Pass Rate')
    plt.ylabel('Model Bad Rate')
    plt.legend(loc = 1)
    
    bad_rate_result.index = np.arange(0.01,1,0.01)
    return bad_rate_result





def plot_density_curve(y_true,y_predict):
    """
    用于绘制不同类的概率分布图 目前只支持2分类 0代表好 1代表坏
    y_true: 真实标签 非概率 Series或者array
    y_predict: 模型预测的结果 概率 Series或者array
    """
    plt.figure(figsize=(8,6))
    sns.kdeplot(y_predict[y_true==0],label='good')
    sns.kdeplot(y_predict[y_true==1],label='bad')
    sns.kdeplot(y_predict,label='prob_well_rate')
    plt.show()

# 绘制PR曲线    
def plot_PR_curve(y_true,y_predict):
    '''
    用户绘制PR曲线图
    y_true: 真实标签 非概率 Series或者array
    y_predict: 模型预测的结果 概率 Series或者array
    '''
    precision,recall,thresholds=precision_recall_curve(y_true,y_predict)
    average_precision = average_precision_score(y_true, y_predict)
    ax2 = plt.subplot()
    ax2.set_title("Precision_Recall Curve AP=%0.2f"%average_precision,verticalalignment='center')
    plt.plot(recall,precision,lw=1)
    plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck")		 
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.show()    


def plot_multi_PR_curve_dict_type(predict_dict):
    """
    用于多个模型绘制PR曲线 并返回每个模型的详细结果
    predict_dict 多个模型的预测结果和真实结果组成的dict 形如{'model1':{'predict':[],'true':[]},.....}
    """
    plt.figure(figsize=(8,6))
    for model_name in predict_dict.keys():
        y_predict_prob = predict_dict.get(model_name).get('predict')
        y_true = predict_dict.get(model_name).get('true')
        precision,recall,thresholds=precision_recall_curve(y_true,y_predict_prob)
        average_precision = average_precision_score(y_true, y_predict_prob)
        plt.plot(recall,precision,lw=1,label=model_name + " Precision_Recall Curve AP=%0.2f"%average_precision)
    plt.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6),label="Luck")
    plt.legend()   		 
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("Recall Rate")
    plt.ylabel("Precision Rate")
    plt.show()

# 绘制验证曲线
def plot_validation_curve(estimator, title, X, y,param_name,param_range,ylim=(0,1),cv=5,scoring='auc'):
    """
    用于绘制验证曲线 并返回每一步的结果
    estimator: 评估的分类器
    title: 图像标题 str
    X: X数据
    y: Y数据
    param_name: 进行评估的参数名称 str
    param_range: 进行评估的参数区间 list or array
    ylim: 图像y轴体上下界 例如(0,1)
    cv: cross-validation参数 参考validation_curve的文档
    scoring: 评估指标 sklearn.metrics.SCORERS所列的评估指标
    
    return: pd.DataFrame
    """
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    return pd.DataFrame({'train_scores_mean':train_scores_mean,'train_scores_std':train_scores_std,\
                         'test_scores_mean':test_scores_mean,'test_scores_std':test_scores_std})

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5),scoring='roc_auc'):
    """
    estimator: 评估的分类器
    title: 图像标题
    X: X数据
    y: Y数据
    y_lim: 图像y轴体上下界
    cv: cross-validation参数 参考learning_curve的文档
    train_sizes: 用于评估的不同训练集尺寸 例：np.linspace(0.1, 1.0, 5)
    scoring: 评估指标 sklearn.metrics.SCORERS所列的评估指标
    
    return: pd.DataFrame
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=6,scoring =scoring, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return pd.DataFrame({'train_scores_mean':train_scores_mean,'train_scores_std':train_scores_std,\
                         'test_scores_mean':test_scores_mean,'test_scores_std':test_scores_std})

    

