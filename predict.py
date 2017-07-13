# -*- coding:utf-8 -*-
import numpy as np
import sklearn
from data import data_trans
from model import logistic
from sklearn.externals import joblib
import pandas as pd
from sklearn import preprocessing
import os
import shutil
import codecs
import logging
import logging.handlers
import datetime

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

def trans_input():
    """
    读取txt文件中的数据转换为dataframe
    :param DataFrame
    :return: excel 文件路径
    """
    path = 'traceback_excel/conflict_model_input_'+datetime.datetime.now().AddDays(-1).strftime('%Y%m%d')+'.txt'
    if(not os.path.exists(path)):
        logger.debug('input file not exist !!!')
        return
    file = codecs.open(path,encoding='utf-8').read().split('\n')
    line_count = 1
    result =  []                #pd.DataFrame(columns=file[0].split('\t'))
    for i in file[1:]:
        temp = i.split('\t')
        if(len(temp)!=7):
            logger.debug('line '+str(line_count)+':column number error！！！')
        else:
            result.append(temp)
        line_count+=1
    logger.info('read txt success!!')
    result = pd.DataFrame(result,columns=file[0].split('\t'))
    logger.info('generate input excel suceess!!!')
    return result


def predict(data,rates,order):
    """
    用指定模型对文件标注
    :param clf_file: 模型文件
    :param rates: 二分类通过率梯度
    :param order: 而分类顺序
    :return: None
    """
    # 类别映射表
    index_dict = [u'乘客醉酒／涉毒／怀疑绑架', u'交通事故致伤', u'人伤', u'人车不符（已涉煤）', u'其他', u'司机酒驾／毒驾',
                  u'失联', u'强奸', u'恐吓', u'抢夺/抢劫', u'损毁财物', u'携带危险物品', u'杀人', u'物品遗失',
                  u'猝死', u'盗窃??', u'绑架', u'群体事件致伤', u'自杀威胁', u'诈骗／敲诈', u'限制自由',
                  u'骚扰／性骚扰']
    result = data
    data = data.iloc[:,[1,3,5,6]]
    # 获取预测数据特征矩阵
    data.columns = ['user_type', 'content', '2nd-title', '3rd-title']

    try:
        data = data_trans.get_matrix(data)
    except:
        logger.debug('load test matrix failed')
    trainX = data
    trainX = preprocessing.scale(trainX)
    py = pd.Series([np.nan]*trainX.shape[0],index=data.index)

    for r in rates:
        for i in order:
            try:
                # 获取模型对应的类别编号
                index = int(i)
                # 获取模型
                clf = joblib.load('model/lr_model/lr-' + i)
                logger.info( 'load model success!!')
                # 对应类识别
                predicty = pd.Series(clf.predict_proba(trainX)[:, 1], index=data.index)
                # 标号转数字
                for i in predicty.index:
                    if (predicty[i] > r and py[i]!=py[i]):
                        py[i] = index_dict[index]
            except:
                logger.debug('class ', i, ' error!!')
    result['predict'] = py
    result.to_csv('traceback_excel/conflict_model_output_'+datetime.datetime.now().strftime('%Y%m%d')+'.txt',index=False,sep='\t',encoding='utf8')
    logger.info('predict end !!!')

if __name__ == "__main__":
    data = trans_input()
    rates = [0.5,0.4,0.3,0.2,0.1,0.04]
    order = ['1', '17', '2', '0', '11', '10', '21', '13', '8', '7', '20', '19', '14', '15', '16', '9', '4']
    predict(data, rates, order)
    # pd.read_excel(filename).to_csv(filename.split('.')[0]+'.txt',index=False,sep='\t',encoding='utf8')
    # shutil.copy('traceback_excel/' + filename, '../test.xlsx')