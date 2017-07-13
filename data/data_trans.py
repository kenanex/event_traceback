# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import cPickle as pickle
import jieba
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import logging
import logging.handlers

def log_init():
    """
    初始化
    :return: logger
    """
    LOG_FILE = 'log/conflict_model.log'
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  # 实例化handler
    fmt = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s'

    formatter = logging.Formatter(fmt)  # 实例化formatter
    handler.setFormatter(formatter)  # 为handler添加formatter

    logger = logging.getLogger('conflict_model')  # 获取名为tst的logger
    logger.addHandler(handler)  # 为logger添加handler
    logger.setLevel(logging.DEBUG)

    return logger
    # logger.info('first info message')  示例1
    # logger.debug('first debug message') 示例2

#logc初始化
logger = log_init()

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

def save_object(object,filename):
	"""
	将数据存入目标文件
	:param object: 数据,all
	:param filename: 目标文件路径，string
	:return: none
	"""
	file=open(filename,'wb')
	pickle.dump(object,file,1)
	file.close()

def data_trans(data,index,missing='normal',onehot=False,word2vec=False,file=''):
	"""
	:param data: 需要转换的数据，DataFrame
	:param index: 需要转换的数据column名，string
	:param missing: 空值处理方式,normal不处理，avg取平均，del删除，onehot哑编码，string
	:param onehot: 对于数字类型是否进行哑编码，bool
	:param word2vec: 对于文本类型是否进行向量转换，bool
	:return:none
	"""
	if(word2vec):
		wmap = get_object(file)
		col = [''] * len(wmap)
		for i in wmap:
			col[wmap[i]] = index + '-' + i
		result = []
		for text in data[index]:
			vec = np.zeros(len(wmap))
			words = jieba.cut(text)
			for w in words:
				if (w in wmap):
					vec[wmap[w]] += 1
			result.append(vec)
		result = pd.DataFrame(result,index=data.index,columns=col)
		del data[index]
		return pd.concat([data,result],axis=1)
	if(missing=='avg'):
		avg = data[index][data[index]!=0].mean()
		data.loc[data[index]==0,index]=avg
	if(onehot):
		tmap = get_object(file)
		col = ['']*len(tmap)
		col.append(index+u'其他')
		for i in tmap:
			col[tmap[i]] = index+i
		result = []
		for i in data[index]:
			vec = np.zeros(len(tmap)+1)
			if(i in tmap):
				vec[tmap[i]] = 1
			else:
				vec[-1] = 1
			result.append(vec)
		result = pd.DataFrame(result,columns=col,index=data.index)
		del data[index]
		return pd.concat([data,result],axis=1)


def get_matrix(data):
	"""
	将指定文件转换为预测特征矩阵
	:param filename:
	:param index_col:
	:param names:
	:return:
	"""
	data = data.dropna()
	data = data_trans(data, 'content', word2vec=True, file='data/map_dict/word_map')
	logger.info('content transform sucess!!')
	data = data_trans(data, '2nd-title', onehot=True, file='data/map_dict/title2_map')
	logger.info( '2nd-title transform sucess!!')
	data = data_trans(data, '3rd-title', onehot=True, file='data/map_dict/title3_map')
	logger.info( '3rd-title transform sucess!!')

	logger.info( 'generate test matrix sucess!!')
	return data


def updata_matrix(index_col,names):
	"""
	更新训练和测试集特征矩阵
	:param index_col: 源文件选用的列号列表
	:param names: 列明制定
	:return:None,实例化矩阵为train_matrix文件
	"""
	print 'read train data...'
	data = pd.read_excel('train_set/train.xlsx', parse_cols=index_col, names=names)			# 读取文件
	print 'load train data sucess!!'
	#print data

	data = data.dropna()
	data = data_trans(data, 'content', word2vec=True, file='map_dict/word_map')
	print 'content transform sucess!!'
	data = data_trans(data, '2nd-title', onehot=True, file='map_dict/title2_map')
	print '2nd-title transform sucess!!'
	data = data_trans(data, '3rd-title', onehot=True, file='map_dict/title3_map')
	print '3rd-title transform sucess!!'

	cmap = get_object('map_dict/class_map')

	for i in data.index:
		if(data.loc[i, 'new-2rd-title'] in cmap):
			data.loc[i, 'new-2rd-title'] = cmap[data.loc[i, 'new-2rd-title']]
		else:
			print 'unrecord class:',data.loc[i, 'new-2rd-title']
			data.loc[i, 'new-2rd-title'] = -1
	print 'generate train matrix sucess!!'

	#进行集合拆分，默认比例9：1
	test = pd.DataFrame()
	for i in range(22):
		temp = data[data['new-2rd-title']==i]
		test = pd.concat([test,temp.sample(int(temp.shape[0]*0.1))])
	data.drop(test.index)

	save_object(data, 'train_matrix')
	save_object(test, 'test_matrix')

if __name__ == "__main__":
	index_cols = [2, 3, 5, 6, 8]
	names = ['user_type', 'content', '2nd-title', '3rd-title','new-2rd-title']
	updata_matrix(index_cols,names)