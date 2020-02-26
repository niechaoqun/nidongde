#encoding:utf8

from sklearn.externals import joblib
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd

from RF_Tree_Extractor import RF_Tree_Extractor

ORG_CLASS_KEY = 'org_class'
ORG_DATA_KEY = 'org_data'
RESULT_KEY = 'result'
PROVED = 'proved'
FAILED = 'failed'

class Experment_Analyses(object):

    def __init__(self, model):
        self._model = model

        if self._model.startswith('gb'):
            self._top_dir = os.path.join('.', self._model)
        else:
            self._top_dir = os.path.join('.', self._model)
        # print(self._top_dir)


        self._clf = None
        self._dts_importance = {}

    def load_model(self):
        model_path = os.path.join(self._top_dir, 'model')
        model_name = os.path.join(model_path, '%s.pkl'%self._model)
        self._clf = joblib.load(model_name)
        self.model_show = RF_Tree_Extractor(estimators=self._clf.estimators_, is_rf=True)
        # print(s_clf.predict(org_data.reshape(1, -1)))
    def show_train_ratio(self, data_path):
        train_ratio = pd.read_csv(data_path)
        print(train_ratio.shape)
        target = train_ratio.iloc[:, -1]
        # print(target.min)
        distribution = Counter(target.tolist())
        print(distribution)

    def unsat_feature_dot_pic(self, feature_name, unsat_data, feature_data, class_name):

        plt.scatter(feature_name, feature_data, marker = '.',color = 'black', s = 15, label = '%s_class_unsat_node' % class_name )

        plt.scatter(feature_name, unsat_data, marker = 'x', color = 'red', s = 15, alpha=0.5, label = 'feature_importance')

        plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
        # plt.show()
        plt.savefig('feature_unsat_%s.jpg' % class_name, dpi=300) # dpi 可以用来设置分辨率
        plt.clf()

    def unsat_node_dot_pic(self, feature_name, unsat_data, class_name):

        plt.scatter(feature_name, unsat_data, marker = 'x', color = 'red', s = 15, alpha=0.5, label = '')

        plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
        # plt.show()
        plt.savefig('unsat_node_%s.jpg' % class_name, dpi=300) # dpi 可以用来设置分辨率
        plt.clf()

    def show_model(self):
        self.model_show.show()

    def show_unsat_node(self, node_Dict):

        feature_names = np.arange(784).reshape(784,)
        feature_importance = self._clf.feature_importances_
        # make importances relative to max importance
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        # print(feature_importance.sum())
        f_sorted_idx = np.argsort(feature_importance) # 按照升序排列
        
        # plt.scatter(feature_names, feature_importance, marker = '.', color = 'red', s = 15, alpha=0.5, label = 'feature_importance')
        # plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
        # plt.show()
        # plt.savefig('%s_feature_importance_dot.jpg' % self._model, dpi=300) # dpi 可以用来设置分辨率
        # plt.clf()

        feature_show_data = np.array(feature_importance).reshape(28, 28)
        # plt.title('model_feature')
        # plt.imshow(feature_show_data)
        # plt.show()
        # plt.savefig('%s_feature_importance_dot.jpg' % self._model, dpi=300) # dpi 可以用来设置分辨率

        for k, v in node_Dict.items():

                # break
                # print("-"*10)
                # print("class:%s"%k)
                nodes = Counter(v)
                tmp = [0]*784
                for n, value in nodes.items():
                    tmp[int(n.replace('x', ''))] = value


                node_cpr_data = np.array(tmp).reshape(784,)
                # node_cpr_data = (node_cpr_data / node_cpr_data.max())
                node_cpr_data_percent = (node_cpr_data / node_cpr_data.sum())
                # node_cpr_data_percent = 100.0 * (node_cpr_data_percent / node_cpr_data_percent.max())
                n_sorted_idx = np.argsort(node_cpr_data)
                reduce_sorted_idx = n_sorted_idx[:30]

                # plt.scatter(feature_names, node_cpr_data_percent, marker = '.', color = 'blue', s = 15, alpha=0.5, label = 'feature_importance')
                # plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
                # # plt.show()
                # plt.savefig('%s_unsat_node_dot.jpg'%k, dpi=300) # dpi 可以用来设置分辨率
                # plt.clf()
                # print(node_cpr_data_percent.sum())

                #根据 node 的出现次数，来显示dot 图
                # node_occurs_show_data = np.array(node_cpr_data).reshape(28, 28)
                # plt.title('class:%s'%k)
                # plt.imshow(node_occurs_show_data)
                # plt.show()
                # plt.savefig('class_%s_unsat_node_dot.jpg'%k, dpi=300) # dpi 可以用来设置分辨率

                # print(feature_importance)
                # print(node_cpr_data)

                # step: 计算unsat_node 重要度和 feature_importance的差值的绝对值之和
                sub_abs =  abs(feature_importance - node_cpr_data_percent)
                # print(feature_importance)
                # print(node_cpr_data_percent)
                print("[%s]%s" % (k , sub_abs.sum()))
                # print("%s" % (sub_abs.sum()/100))

                # # step: 计算unsat_node 重要度和 各个基学习器feature_importance的差值的绝对值之和
                # for dt, f_i in self._dts_importance.items():

                #     dt_abs =  abs(f_i/100 - node_cpr_data/100)
                #     # print("[%s]%s" % (k , sub_abs.sum()/100))
                #     print("tree_%s:%s" % (dt, dt_abs.sum()/100))
                
                # step:显示 unsat_node 和 feature_importance 的组合图
                self.unsat_feature_dot_pic(feature_names, n_sorted_idx, f_sorted_idx, k)
                
                # step:单独显示 unsat_node 图
                # self.unsat_node_dot_pic(feature_names, n_sorted_idx, k)

                # break
                

    def get_clf_dt_importance(self):
        """
          显示模型中 各个基学习器的 feature_importance 特征
        """
        i = 0
        for estimator in self._clf.estimators_:
            print('tree_%s'% i)
            feature_names = np.arange(784).reshape(784,)
            feature_importance = estimator.feature_importances_
            # make importances relative to max importance
            feature_importance = 100.0 * (feature_importance / feature_importance.max())
            f_sorted_idx = np.argsort(feature_importance)

            plt.scatter(feature_names, f_sorted_idx, marker = '.',color = 'blue', s = 15, label = 'tree_%s_importance' % i )
            plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
            plt.savefig('tree_%s_feature_importance.jpg' % i, dpi=300) 
            plt.clf()
            self._dts_importance[str(i)] = feature_importance
            i+=1

    def check_node_pertubition(self, org_data, unsat_node_list):
        """
          测试 unsat_list 中的节点，对扰动值的敏感程度
        """

        data = np.array(org_data)
        tmp1 = data.copy()
        tmp2 = data.copy()
        unsat_index_list = [int(n.replace('x', '')) for n in unsat_node_list]
        for pix in range(255):
        # for pix in [255]:
            for i in range(data.shape[0]):
                if i not in unsat_index_list:
                    tmp1[i] += pix
                    pass
                else:
                    # tmp1[i] += pix
                    pass
            show_data1 = np.array(tmp1).reshape(28, 28)
            plt.title('not in unsat:%s'%self._clf.predict(tmp1.reshape(1, -1)))
            plt.imshow(show_data1)
            plt.show()
            plt.clf()
            if self._clf.predict(tmp1.reshape(1, -1))[0] !=0:
                print("[%s]not in unsat predict:%s"%(pix, self._clf.predict(tmp1.reshape(1, -1))[0]))
                break
        # print("in unsat predict:%s"%self._clf.predict(tmp2.reshape(1, -1)))
        
        # for i in range(data.shape[0]):
        #     tmp = data.copy()
        #     tmp[i] +=255
        #     # print("replace predict:%s"%self._clf.predict(tmp.reshape(1, -1))) 
        #     prf = self._clf.predict(tmp.reshape(1, -1))[0]
        #     # print()
        #     if prf !=4:
        #         print("replace predict:%s" % prf)
        #         print(i)
        #         continue   



        # show_data2 = np.array(tmp2).reshape(28, 28)
        # plt.title('in unsat:%s'%self._clf.predict(tmp2.reshape(1, -1)))
        # plt.imshow(show_data2)
        # plt.show()
        # plt.clf()
 
         
    def check_predict_result(self, org_data):
        org_predict = {}
        data = np.array(org_data).reshape(1, -1)
        i=0
        for estimator in self._clf.estimators_:
            # print('tree_%s'% i)
            value = estimator.tree_.value
            result = estimator.predict(data)[0]
            proba = estimator.predict_proba(data)[0]
            print("proba:%s"%proba)
            leaf = estimator.apply(data)[0]
            # print("decision_path:%s" % estimator.decision_path(data))
            print("leaf:%s value:%s" % (leaf, value[leaf]))
            org_predict.setdefault(result, 0)
            org_predict[result] +=1
            i+=1
        print("org_predict:")    
        print(org_predict)

    def analyse_verify_result(self):

        # feature_importance = self._clf.feature_importances_
        # print(feature_importance.shape)

        result_path = os.path.join(self._top_dir, 'result')

        proved_num = 0
        total_num = 0

        tmp_file = []
        unsat_tree_dict = {}
        unsat_node_dict = {}
        proved_dict = {}  # 记录该类别的 proved 的样本数
        class_example_num = {} #记录该类别总验证样本数
        num_data = 0    
        for f in os.listdir(result_path):
            # print('-'*20)
            # if f.endswith('.json'):
               
            if f.endswith('.json') and f.startswith('0_verify_0'):
             # and f.startswith('0_verify'):

                # print('file:%s'% f)
                filePath = os.path.join(result_path, f)
                
                resultFile = open(filePath, 'r')
                result_dict = json.load(resultFile)

                org_class = result_dict[ORG_CLASS_KEY]
                org_data = result_dict[ORG_DATA_KEY]

                # self.check_predict_result(org_data)
                org_predict = self._clf.predict(np.array(org_data).reshape(1, -1))[0]

                # 去除原模型识别不正确的样本
                if org_predict != org_class:
                    continue
                
                class_name = f.split('_')[0]  
                proved_dict.setdefault(class_name, 0)
                class_example_num.setdefault(class_name, 0)
  
                total_num += 1 
                class_example_num[class_name] +=1
                # break
                verify_result = result_dict[RESULT_KEY]

                num_data = len(org_data)
     
                if PROVED == verify_result:
                    # print('class:%s'%f.split('_')[0])
                    # print("Property: " + PROVED)
                    proved_num +=1
                    unsat_list = result_dict['unsat']
                    
                    unsat_tree_list = []
                    unsat_node_list = []
                    robust_data = [0]* num_data
                    for item in unsat_list:
                        if item.startswith('bx'):
                            robust_data[int(item.replace('bx', ''))] = org_data[int(item.replace('bx', ''))]
                            unsat_node_list.append(item.replace('bx', 'x'))
                        if item.startswith('tree'):
                            unsat_tree_list.append(item)

                    show_data1 = np.array(robust_data).reshape(28, 28)
                    plt.title('in unsat:%s'%self._clf.predict(show_data1.reshape(1, -1)))
                    plt.imshow(show_data1)
                    plt.show()
                    plt.clf()

                    # 测试 unsat_list中的节点，对鲁棒性的影响
                    self.check_node_pertubition(org_data, unsat_node_list)

                    sample = np.array(robust_data).reshape(28, 28)
                    
                    unsat_tree_dict.setdefault(class_name, [])
                    unsat_tree_dict[class_name].extend(unsat_tree_list)

                    unsat_node_dict.setdefault(class_name, [])
                    unsat_node_dict[class_name].extend(unsat_node_list)

                    proved_dict[class_name] += 1
                    
                    org_predict = self._clf.predict(np.array(org_data).reshape(1, -1))[0]
                    # print("org:%s" % (org_predict))
                    # break

                elif FAILED == verify_result:
                    # print("Property: " + FAILED)
                    
                    org_add_smt_data = org_data[:]

                    tst_data = []
                    smt_data = [0]*len(org_data)
                    wl_data = {}
                    for k, v in result_dict.items():
                        if k.startswith('x'):
                            smt_data[int(k.replace('x', ''))] = int(v)
                            if int(v) != 0:
                                tst_data.append(int(v))
                            org_add_smt_data[int(k.replace('x', ''))] = int(v)
                        elif k.startswith('wl_'):
                            index = k.split('_')[-1]
                            # print(index)
                            wl_data.setdefault(index, 0)
                            # wl_data[index] += int(v)
                    
                    smt_predict = self._clf.predict(np.array(smt_data).reshape(1, -1))[0]
               
                    
        # for k, v in unsat_tree_dict.items():
        #     tree = Counter(v)
        #     print('%s unsat tree' % k)
        #     print(tree)

        #     nodes = Counter(unsat_node_dict[k])
        #     print('%s unsat node' % k)
        #     print(nodes)

        # 处理 各个类型的 unsat 特征
        # self.show_unsat_node(unsat_node_dict)

        # 统计不同类别的 鲁棒性
        # for k, v in proved_dict.items():
        #     base = class_example_num[k]
        #     ratio = v/base
        #     # print("[%s] total_num:%s robust ratio:%s" %(k, class_example_num[k], v/class_example_num[k]*()))
        #     print("[%s] total_num:%s robust ratio:%s" %(k, 100, ratio))

        # 统计总体模型的鲁棒性
        # print("total sample num:{}".format(total_num))
        # print('proved ratio:{}'.format(proved_num /total_num))


if __name__ == "__main__":
    for model in ['rf_mnist_20_3_1_robust']:
    # for model in ['rf_mnist_50_8_1_robust']:
        ea = Experment_Analyses(model)
        ea.load_model()
        # ea.show_model()
        # ea.get_clf_dt_importance()
        ea.analyse_verify_result()
    


    # ea.show_train_ratio('./rf_mnist_50_8_1_robust/model/rf_mnist_50_8_1_robust.csv')
        # ea.show_train_ratio('rf_mnist_20_3_1_robust/model/rf_mnist_20_3_1_robust.csv')
    # ea.show_train_ratio('sklearn_rf/rf_mnist_20_8_1_robust_ib/model/rf_mnist_20_8_1_robust_ib.csv')
    # ea.show_train_ratio('sklearn_rf/datasets/mnist_train_1.0.csv')

    # mnist = pd.read_csv('./datasets/mnist.csv')
