# -*- coding: utf-8 -*-
import os
import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
import xgboost as xgb
from sklearn.externals import joblib
from sklearn import cross_validation,metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV as logisticCV
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
rcParams['font.sans-serif'] = ['SimHei']

def get_object(filename):
	"""
	从目标文件中获取数据
	:param filename: 数据类型的目标文件路径,string
	:return: 数据
	"""
	file=open(filename,'rb')
	result=pickle.load(file)
	file.close()
	return result


def model_report(index):
    """
    生成制定标号对应类别的模型报告
    :param index: 类别标号
    :return: None
    """
    model = joblib.load('lr_model/lr-'+str(index))
    data = get_object('../data/test_matrix')
    data.loc[data['new-2rd-title'] != index, 'new-2rd-title'] = -1
    data.loc[data['new-2rd-title'] == index, 'new-2rd-title'] = 1
    data.loc[data['new-2rd-title'] == -1, 'new-2rd-title'] = 0

    trainY = data['new-2rd-title'].values.astype('int')
    del data['new-2rd-title']
    trainX = data.values
    trainX = preprocessing.scale(trainX)

    dtrain_predictions = model.predict(trainX)
    dtrain_predprob = model.predict_proba(trainX)[:, 1]

    # Print model report:
    print "\nModel Report"
    print "accuracy_score : %.4g" % metrics.accuracy_score(trainY, dtrain_predictions)
    print "precision_score: %.4g"% metrics.precision_score(trainY, dtrain_predictions)
    print "recall_score: %.4g"% metrics.recall_score(trainY, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(trainY, dtrain_predprob)


def binary_classification(index):
    """
    对指定编号的类别训练xgboost二分类模型，并进行grid-search调参，保存模型至xgb_model.
    :param index: 类别编号
    :return:None
    """
    # index_dict = [u'乘客醉酒／涉毒／怀疑绑架', u'交通事故致伤', u'人伤', u'人车不符（已涉煤）', u'其他', u'司机酒驾／毒驾',
    #    u'失联', u'强奸', u'恐吓', u'抢夺/抢劫', u'损毁财物', u'携带危险物品', u'杀人', u'物品遗失',
    #    u'猝死', u'盗窃??', u'绑架', u'群体事件致伤', u'自杀威胁', u'诈骗／敲诈', u'限制自由',
    #    u'骚扰／性骚扰']          类别对应list

    data = get_object('../data/train_matrix')
    data.loc[data['new-2rd-title'] != index,'new-2rd-title'] = -1
    data.loc[data['new-2rd-title'] == index,'new-2rd-title'] = 1
    data.loc[data['new-2rd-title'] == -1, 'new-2rd-title'] = 0

    trainY = data['new-2rd-title'].values.astype('int')

    del data['new-2rd-title']
    trainX = data.values
    trainX = preprocessing.scale(trainX)
    print 'load train matrix sucess!!'

    # lr =logisticCV(Cs=[0.001, 0.1, 10, 100, 10000], penalty='l2')
    lr = logisticCV(Cs=[0.005, 0.008, 0.01, 0.1, 1, 10], penalty='l2')
    lr.fit(trainX, trainY)
    joblib.dump(lr, 'lr_model/lr-'+str(index))
    print 'mean-accuracy:',pd.DataFrame(lr.scores_[1]).mean(axis=0)
    print 'best_params:',lr.C_


if __name__ == "__main__":
	#model_report(0)
    for i in range(0, 22):
        try:
            binary_classification(i)
            model_report(i)
        except:
            print 'fail in ', i, 'probabily count less!!'
    # print pd.DataFrame(clf.coef_,columns=cs).T.sort_values(by=[0],ascending=[0])
    # clf = logisticCV(Cs=[0.005, 0.008, 0.01, 0.1, 1, 10, 100], penalty='l2')
    # clf.fit(trainX, trainY)
    # joblib.dump(clf, 'model/LR_l2')
    # print 'l2:'
    # print pd.DataFrame(clf.scores_[1]).mean(axis=0)
    # print clf.C_
    # show_result(clf,feature_dict)


    # ll = len(py)
    # p1 = 0
    # p0 = 0
    # for i in range(ll):
    #     if(py[i]==1):
    #         if(trainY[i]==1):
    #             p0+=1
    #         p1+=1
    # print '准确率（人伤）：',p0*1.0/p1
    # p1 = 0
    # p0 = 0
    # for i in range(ll):
    #     if (trainY[i] == 1):
    #         if (py[i] == 1):
    #             p0 += 1
    #         p1 += 1
    # print '召回率（人伤）：',p0 * 1.0 / p1