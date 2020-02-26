# encoding=utf-8

from sklearn.externals import joblib
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from RF_Tree_Extractor import  RF_Tree_Extractor
from RF_SMT_Z3_Cst_Formulas import RF_SMT_Z3
from SMT_Result_Processer import Verify_Result_Processer

class RF_Main_Process(object):
    def __init__(self, model_name, is_rf, epsilon=1, clf=None, test_x=None, tesy_y=None):

        self._top_dir = os.getcwd()

        self._model_name = model_name
        self._verify_dir = '%s/%s/%s' % (self._top_dir, self._model_name, 'verify')
        self._result_dir = '%s/%s/%s' % (self._top_dir, self._model_name, 'result')
        self._model_dir = '%s/%s/%s' % (self._top_dir, self._model_name, 'model')
        self._pic_dir = '%s/%s/%s' % (self._top_dir, self._model_name, 'pic')
        self._prepare_dirs()

        self._model_path = '%s/%s.pkl' % (self._model_dir, self._model_name)
        self._clf = clf

        self._test_x = test_x
        self._tesy_y = tesy_y

        self._epsilon = epsilon

        self._formular_extracter = RF_Tree_Extractor(self._clf.classes_, self._epsilon, clf.estimators_, is_rf)
        
        self._smt = RF_SMT_Z3(isRegression=False, verify_dir=self._verify_dir, result_dir=self._result_dir)

        self._rst_processor = Verify_Result_Processer(self._pic_dir)
   
    def save_train_data_to_csv(self, train_data):
        train_data.to_csv('%s/%s.csv' % (self._model_dir, self._model_name))

    def save_feature_importance(self, acc):

        feature_names = np.array([ i.replace('pixel', 'x') for i in self._test_x.columns.tolist()])
        # Plot feature importance
        feature_importance = self._clf.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)

        reduce_sorted_idx_list = []
        for i in sorted_idx:
            if feature_importance[i] >0:
                reduce_sorted_idx_list.append(i)
        tmp = reduce_sorted_idx_list[:30]

        reduce_sorted_idx = np.array(tmp).reshape(len(tmp),)
        # print(reduce_sorted_idx)
        # print(type(reduce_sorted_idx))
        pos = np.arange(reduce_sorted_idx.shape[0]) + .5

        plt.barh(pos, feature_importance[reduce_sorted_idx], align='center')
        plt.yticks(pos, feature_names[reduce_sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        # plt.show()
        plt.savefig('%s/%s.jpg' % (self._model_dir, self._model_name))

        with open('%s/%s_feature_importance.txt' % (self._model_dir, self._model_name), 'w') as f:
            f.write('model_acc:%s\n' % acc)
            for i in sorted_idx:
                f.write('%s:%s\n' % (feature_names[i], feature_importance[i]))


    def process_digits(self):

        self.save_model()
        for i in range(10):
            t_x = self._test_x[i].reshape(1, -1)
            t_y = self._tesy_y[i]

            self._formular_extracter.set_sample(t_x)
            regression_formulas = self._formular_extracter.get_decision_tree_formulas(t_x)
            org_predict = self._clf.predict(t_x)[0]

            self._smt.set_test_sample_info(org_data=t_x, org_class=t_y, epsilon=self._epsilon, model_name=self._model_name)
            
            self._smt.solve_formula_counter(regression_formulas, org_predict)

            verify_file = self._smt.get_smt_verify_pyFile_name()

            os.system('python3 %s'%verify_file)

            result_file = self._smt.get_smt_verify_result_file()
            
            self._rst_processor.set_info(result_file, 8)

            self._rst_processor.parse_smt_resultFile(self._pic_dir)


    def process_mnist_robust(self, is_reduce, acc):

        self.save_model()

        self.save_feature_importance(acc)

        for i in range(len(self._test_x)):
        # for i in [82, 94, 61, 77, 98, 20, 99, 60, 9, 95, 67, 26, 84, 47, 11, 50, 93, 70, 66, 0, 73, 69, 87, 91, 29, 64, 43, 96, 80, 59, 62, 74, 7, 81, 97, 54]:
        # for i in [82]:
            
            t_x = self._test_x.iloc[i, :].values.reshape(1, -1)
            t_y = self._tesy_y.iloc[i]

            self._formular_extracter.set_sample(t_x)
            self._formular_extracter.set_is_add_reduce(is_reduce)
            regression_formulas = self._formular_extracter.get_decision_tree_formulas(t_x)

            org_predict = self._clf.predict(t_x)[0]

            self._smt.set_test_sample_info(org_data=t_x, org_class=t_y, epsilon=self._epsilon, model_name=self._model_name)
            self._smt.solve_formula_robust(regression_formulas, org_predict)

            verify_file = self._smt.get_smt_verify_pyFile_name()

            os.system('python3 %s'%verify_file)

            result_file = self._smt.get_smt_verify_result_file()
            
            self._rst_processor.set_info(result_file, 28)

            self._rst_processor.parse_robust_resultFile(org_predict)
            # break

    def process_mnist_counter(self):
        self.save_model()
        self.save_feature_importance()

        # for i in range(len(self._test_x)):
        for i in range(len(self._test_x)):
            t_x = self._test_x.iloc[i, :].values.reshape(1, -1)
            t_y = self._tesy_y.iloc[i]

            self._formular_extracter.set_sample(t_x)
            regression_formulas = self._formular_extracter.get_decision_tree_formulas(t_x)

            model_predict = self._clf.predict(t_x)[0]
            org_predict = t_y

            self._smt.set_test_sample_info(org_data=t_x, org_class=t_y, epsilon=self._epsilon, model_name=self._model_name)
            self._smt.solve_formula_counter(regression_formulas, org_predict)

            verify_file = self._smt.get_smt_verify_pyFile_name()

            os.system('python3 %s'%verify_file)

            result_file = self._smt.get_smt_verify_result_file()
            
            self._rst_processor.set_info(result_file, 28)

            self._rst_processor.parse_counter_resultFile(model_predict)


    def save_model(self):
        print("save model pkl:%s" % self._model_path)
        joblib.dump(self._clf, self._model_path)


    def _prepare_dirs(self):
        if not os.path.exists(self._model_name):
            os.mkdir(self._model_name)

        if not os.path.exists(self._verify_dir):
            os.mkdir(self._verify_dir)

        if not os.path.exists(self._result_dir):
            os.mkdir(self._result_dir)

        if not os.path.exists(self._model_dir):
            os.mkdir(self._model_dir)

        if not os.path.exists(self._pic_dir):
            os.mkdir(self._pic_dir)

        




        