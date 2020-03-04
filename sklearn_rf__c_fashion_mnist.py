from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


import numpy as np
import pandas as pd
import time

from RF_Main_Process import RF_Main_Process


if __name__ == "__main__":

    fashion_mnist = pd.read_csv('./datasets/fashion_mnist.csv')
    X = fashion_mnist.iloc[:, 0:-1]
    y = fashion_mnist.iloc[:, -1]

    train_samples = 10000

    # # 划分训练集 和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_samples, test_size=1000)

    for j in [20]:
        for i in [3, 8]:
            for e in [1]:
                num_estimators = j
                num_max_depth = i
                robust_epsilon = e
                model_name = 'rf_fashion_mnist_%s_%s_%s_robust' % (num_estimators, num_max_depth, robust_epsilon)

                clf = RandomForestClassifier(n_estimators=num_estimators,
                                             max_depth=num_max_depth).fit(X_train, y_train)
                # clf = RandomForestClassifier(n_estimators=num_estimators,
                #                              max_depth=num_max_depth, random_state=200).fit(X, y)
                # X_train['class'] = y_train
                # if num_max_depth == 8:
                mnist_test = pd.read_csv('./datasets/fashion_mnist_test.csv')

                verify_x = mnist_test.iloc[:, 0:-1]
                verify_y = mnist_test.iloc[:, -1]

                # model_path = './{0}/model/{0}.pkl'.format('rf_mnist_%s_%s_1_robust' % (num_estimators, num_max_depth))
                # else:
                # model_path = '%s.pkl'%'./rf_fashion_mnist_20_8_1_robust/model/rf_fashion_mnist_20_8_1_robust'
                
                # clf = joblib.load(model_path)

                score = clf.score(verify_x, verify_y)
                # score = clf.score(X_test, y_test)

                print("model_score:%s" % score)

                start = time.time()
                is_rf = True

                main = RF_Main_Process(model_name, is_rf, robust_epsilon, clf, verify_x, verify_y)

                main.save_train_data_to_csv(X_train)
                # # # # # 处理显示反例unsatcore
                # # # main.process_mnist_counter()

                main.save_train_data_to_csv(X_train)
                is_recude = False
                # for class_num in range(10):
                #     main.process_mnist_r  obust(class_num, is_recude, score)
                main.process_mnist_robust_test(is_recude, score)
                # 验证鲁棒性

                end = time.time()
                print("totoal time:%s s" % (end-start))
