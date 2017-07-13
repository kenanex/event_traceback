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

def modelfit(alg, trainX, trainY, tree_num=True, cv_folds=5, early_stopping_rounds=10):
	"""
	网格搜索调参生成最优模型
	:param alg: 原模型
	:param trainX: 训练特征
	:param trainY: 训练标签
	:param tree_num: 是否对迭代层数调优
	:param cv_folds: 交叉验证叠数
	:param early_stopping_rounds:当tree_num为True时生效，设置每隔多少轮判断推出条件
	:return: 调优后的模型
	"""
	if tree_num:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(trainX, label=trainY)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
						  metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=True)
		alg.set_params(n_estimators=cvresult.shape[0])
		print 'the best n_estimators：',cvresult.shape[0]
	param_grid = {
		'max_depth': range(3, 10, 2),
		'min_child_weight': range(1, 6, 2)
	}
	gsearch1 = GridSearchCV(estimator=alg,param_grid=param_grid, scoring='accuracy', n_jobs=4, iid=False, cv=5)
	gsearch1.fit(trainX, trainY)
	print 'best_params:',gsearch1.best_params_
	print 'Auccuracy:',gsearch1.best_score_

	model = gsearch1.best_estimator_
	return model

def model_report(index,feature_eval=False):
	"""
	生成制定标号对应类别的模型报告
	:param index: 类别标号
	:return: None
	"""
	model = joblib.load('xgb_model/xgb-'+str(index))
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

	# print model arguments
	print "best tree_n_estimators:",model.get_xgb_params()['n_estimators']
	print "best max_depth:", model.get_xgb_params()['max_depth']
	print "best min_child_weight:", model.get_xgb_params()['min_child_weight']

	# Print model report:
	print "\nModel Report"
	print "accuracy_score : %.4g" % metrics.accuracy_score(trainY, dtrain_predictions)
	print "precision_score: %.4g" % metrics.precision_score(trainY, dtrain_predictions)
	print "recall_score: %.4g" % metrics.recall_score(trainY, dtrain_predictions)
	print "AUC Score (Train): %f" % metrics.roc_auc_score(trainY, dtrain_predprob)

	# feature eval
	if(feature_eval):
		feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
		x_label = []
		for i in feat_imp.index:
			x_label.append(data.columns[int(i[1:])])
		feat_imp.index = x_label
		feat_imp.plot(kind='bar', title='Feature Importances',)
		plt.ylabel('Feature Importance Score')
		plt.show()

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

	xgb1 = XGBClassifier(
		 n_estimators=200,
		 max_depth=5,
		 min_child_weight=1,
		 objective= 'binary:logistic',
		 scale_pos_weight=1,
		 seed=27)

	model = modelfit(xgb1,trainX,trainY)
	joblib.dump(model, 'xgb_model/xgb-' +str(index))


if __name__ == "__main__":
	print get_object('../data/train_matrix')
	#model_report(0)
	# for i in range(0,22):
	# 	try:
	# 		# binary_classification(1)
	# 		model_report(i)
	# 	except:
	# 		print 'fail in ',i,'probabily count less!!'
