#encoding=utf-8
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from z3 import *
import pickle
import json
import os


import pandas as pd
import time
from sklearn.model_selection import train_test_split
from RF_Main_Process import RF_Main_Process
from RF_Tree_Extractor import RF_Tree_Extractor


def get_abs_min_max(epsilon, original_data):
    min_num = original_data-epsilon
    max_num = original_data+epsilon

    if min_num <=0:
        min_num = 0
    return min_num, max_num

 
if __name__ == "__main__":


    # robust_data = [0]* 784
    # # for pix in unsat_list:
    # robust_data[578] = 255 
    # robust_data[430] = 255 
    # robust_data[356] = 255 
    # robust_data[570] = 255 
    # robust_data[378] = 255 
    # robust_data[631] = 255 
    # robust_data[543] = 255 
    # robust_data[596] = 255 
    # robust_data[382] = 255 
    # robust_data[376] = 255 
    # robust_data[596] = 255 
    # robust_data[213] = 255 
    # robust_data[386] = 255 
    # sample = np.array(robust_data).reshape(28, 28)
    # plt.imshow(sample)
    # plt.show()

    # print(get_abs_min_max(1, 0))


    # model_name = '%s.pkl'%'/Users/rose/PycharmProjects/dtVerify/sklearn_rf/rf_mnist_20_3_1/model/rf_mnist_20_3_1'

    # clf = joblib.load(model_name)

    # new  = clf.estimators_[0:5]

    # clf.estimators_ = new

    # rt = RF_Tree_Extractor(clf.classes_, 1, clf.estimators_, True)
    # rt.show()

 # 准备数据集的代码
    mnist = pd.read_csv('./datasets/mnist.csv')
    print(mnist.shape)

    test_data = []
    for i in range(10):
        # for i in 
        lable = mnist[mnist['class']==i]
        if i==1 or i==4 or i==8 or i==6 or i==2:
            l = lable.iloc[500:1500]
        else:
            l = lable.iloc[500:1000]
        test_data.append(l)
    
    result = pd.concat(test_data)
    print(result)
    result.to_csv("datasets/mnist_train_2.0.csv", encoding="utf-8-sig", header=True, index=False)
######  end