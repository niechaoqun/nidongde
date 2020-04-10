#encoding:utf8

from sklearn.externals import joblib
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import time
import shutil

from RF_Tree_Extractor import RF_Tree_Extractor

ORG_CLASS_KEY = 'org_class'
ORG_DATA_KEY = 'org_data'
RESULT_KEY = 'result'
PROVED = 'proved'
FAILED = 'failed'
TIMEOUT = 'timeout'

FASHION_CLASS_DIC = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
def show_dot_pic(clf_1, clf_2, test_data):

    feature_names = np.arange(784).reshape(784,)
    feature_importance_1 = clf_1.feature_importances_
    feature_importance_2 = clf_2.feature_importances_
    
    # plt.scatter(feature_names, feature_importance_1, marker = '.', color = 'red', s = 15, alpha=0.5, label = 'feature_importance_1')
    # plt.scatter(feature_names, feature_importance_2, marker = '.', color = 'blue', s = 15, alpha=0.5, label = 'feature_importance_2')
    # plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
    # # plt.show()
    # plt.savefig('feature_importance_cmp.jpg', dpi=300) # dpi 可以用来设置分辨率
    # plt.clf()
    base_data = [0]*784
    for i in range(len(test_data)):
        data = test_data.iloc[i, :].tolist()
        for j in range(784):
        # 如果该节点出现过的话，就+1
            if data[j] > 0:
                base_data[j] +=1

    node_cpr_data = np.array(base_data).reshape(784,)
    # node_cpr_data = (node_cpr_data / node_cpr_data.max())
    node_cpr_data_percent = (node_cpr_data / node_cpr_data.sum())

    sub_abs1 =  abs(feature_importance_1 - node_cpr_data_percent)
    print("[%s]%s" % (1 , sub_abs1.sum()))
    sub_abs2 =  abs(feature_importance_2 - node_cpr_data_percent)
    print("[%s]%s" % (2 , sub_abs2.sum()))



class Experment_Analyses(object):

    def __init__(self, model=None):
        self._model = model

        if self._model.startswith('gb'):
            self._top_dir = os.path.join('./data_back', self._model)
            # self._top_dir = os.path.join('.', self._model)
        else:
            self._top_dir = os.path.join('./data_back', self._model)
            # self._top_dir = os.path.join('.', self._model)
        # print(self._top_dir)

        self._clf = None
        self._dts_importance = {}

        self.distance = []
        self.model_show = None

        self.verfiy_time_result = ''

    def set_model(self, model_name):
        top_dir = os.path.join('.', model_name)
        model_path = os.path.join(top_dir, 'model')
        model_name = os.path.join(model_path, '%s.pkl'% model_name)
        self._clf = joblib.load(model_name)

    def load_model(self):
        model_path = os.path.join(self._top_dir, 'model')
        model_name = os.path.join(model_path, '%s.pkl'%self._model)
        self._clf = joblib.load(model_name)
        # return self._clf
        self.model_show = RF_Tree_Extractor(estimators=self._clf.estimators_, is_rf=True)
        # print(s_clf.predict(org_data.reshape(1, -1)))
    def show_train_ratio(self, data_path):
        train_ratio = pd.read_csv(data_path)
        # print(train_ratio.shape)
        test = train_ratio.iloc[:, :-1]
        return test
        # distribution = Counter(target.tolist())
        # print(distribution)


    def prepare_verify_time_data(self):
        self._top_dir = os.path.join('./verify_time_3', self._model)
        verify_dir = os.path.join(self._top_dir, 'time')

        if not os.path.exists(verify_dir):
            os.mkdir(verify_dir)        

        # test_data= []
        start = time.time()
        num =0

        result_path = os.path.join(self._top_dir, 'result')
        check_path = os.path.join(self._top_dir, 'verify')

        file_list1 = []
        file_list2 = []
        # for num in range(10):
        # for f in os.listdir(verify_dir):
          # 准备测试数据  
        for f in os.listdir(result_path):
            if f.endswith('.json'):
                # filename = f.replace('.py', '.json')
                filePath = os.path.join(result_path, f)
                
                resultFile = open(filePath, 'r')
                result_dict = json.load(resultFile)
                verify_result = result_dict[RESULT_KEY]
                if PROVED == verify_result:
                    # print(verify_result)
                    if len(file_list2) < 6:
                        print(f)
                        file_list2.append(f)
                if FAILED == verify_result:
                    if len(file_list1) < 6:
                        print(f)
                        file_list1.append(f)

                if len(file_list1) >5 and len(file_list2) > 5:
                    file_list1.extend(file_list2)
                    for item in file_list1:
                        filename = item.replace('.json', '.py',)
                        src = os.path.join(check_path, filename)
                        dst = os.path.join(verify_dir, filename)
                        shutil.copyfile(src, dst)
                    break


        for f in os.listdir(verify_dir):
            print('-'*20)
            # 为了验证时间去除无关代码
            if f.endswith('.py'):
                content = ""
                file_path = os.path.join(verify_dir, f)
                with open(file_path, 'r+') as v:
                    content = v.read()
                    content = content.replace('with open(', '# with open(')
                    content = content.replace('f.write(json_str)', '#f.write(json_str)')
                    content = content.replace("json_dict['org_class']", "#json_dict['org_class']")
                    content = content.replace("json_dict['org_data']", "#json_dict['org_data']")
                    content = content.replace("""if s.check() == unsat:
    unsat_list = []
    for i in s.unsat_core():
        unsat_list.append('%s' % i)
    json_dict['unsat'] = unsat_list
    json_dict['result']='proved'
elif s.check() == sat:
    m = s.model()
    json_dict['result']='failed'
    for d in m.decls():
        json_dict[d.name()] = '%s'%m[d]
else:
    json_dict['result']='timeout'
json_str = json.dumps(json_dict, sort_keys=True, indent=4)""","")
                with open(file_path, 'w') as v:
                    v.write(content)

    def calculate_verify_time(self):
        self._top_dir = os.path.join('./verify_time_3', self._model)
        verify_dir = os.path.join(self._top_dir, 'time')
        # test_data= []
        start = time.time()
        num =0

        result_path = os.path.join(self._top_dir, 'result')
        check_path = os.path.join(self._top_dir, 'verify')

        file_list1 = []
        file_list2 = []
        # for num in range(10):
        for f in os.listdir(verify_dir):
           if f.endswith('.py'):
                 # print(f)
                 file_path = os.path.join(verify_dir, f)
                 os.system('python3 %s'%file_path)
                 num +=1

        end = time.time()
        avege_time = '\n{}{}'.format(self._model, (end-start)/num)

        self.verfiy_time_result += avege_time
        print("totoal time:{} s".format((end-start)/num))

    def save_verify_time_result(self):
        with open('verify_time.txt','a') as v:
            v.write(self.verfiy_time_result)

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

    def calculate_base_and_mfi(self, class_name, base_data):
        feature_names = np.arange(784).reshape(784,)
        avege = []
        for idx, tfi in self._dts_importance.items():
            # print("Tree:%s" % idx)
            # feature_importance = self._clf.feature_importances_
            node_cpr_data = np.array(base_data).reshape(784,)
            # node_cpr_data = (node_cpr_data / node_cpr_data.max())
            node_cpr_data_percent = (node_cpr_data / node_cpr_data.sum())
            sub_abs1 =  abs(tfi - node_cpr_data_percent)
            
            avege.append(sub_abs1.sum())
        print("[%s]%s" % (class_name , np.mean(avege)))
        
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
        # # plt.savefig('%s_feature_importance_dot.jpg' % self._model, dpi=300) # dpi 可以用来设置分辨率
        # plt.clf()

        # 显示 feature_importance 的 dot 图
        # feature_show_data = np.array(feature_importance).reshape(28, 28)
        # plt.title('model_feature')
        # plt.imshow(feature_show_data)
        # # plt.show()
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
                node_cpr_data_percent = (node_cpr_data / node_cpr_data.max())
                # node_cpr_data_percent = 100.0 * (node_cpr_data_percent / node_cpr_data_percent.max())
                n_sorted_idx = np.argsort(node_cpr_data)
                reduce_sorted_idx = n_sorted_idx[:30]

                # plt.scatter(feature_names, node_cpr_data_percent, marker = '.', color = 'blue', s = 15, alpha=0.5, label = 'feature_importance')
                # plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
                # # plt.show()
                # plt.savefig('%s_unsat_node_dot.jpg'%k, dpi=300) # dpi 可以用来设置分辨率
                # plt.clf()
                # print(node_cpr_data_percent.sum())

                #根据 node 百分比，来显示dot 图
                fig, ax1 = plt.subplots()

                node_occurs_show_data = np.array(node_cpr_data_percent).reshape(28, 28)

                np.save('fashion_mnist_class_7_rfi', node_occurs_show_data)

                print(node_occurs_show_data)
                ax1.set_title('class:%s' % k)
                pos = ax1.imshow(node_occurs_show_data, cmap='viridis',interpolation='none')
                ax1.set_xticks([])
                ax1.set_yticks([])
                # pos = ax1.imshow(node_occurs_show_data)
                # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace = 0.2)

                # cax = plt.axes([0.82, 0.1, 0.05, 0.8])
                fig.colorbar(pos, ax=ax1)

                plt.show()
                # plt.savefig('class_%s_unsat_node_dot.jpg'%k, dpi=300) # dpi 可以用来设置分辨率
                plt.clf()

                # step: 计算unsat_node 重要度和 feature_importance的差值的绝对值之和
                # sub_abs =  abs(feature_importance - node_cpr_data_percent)
                # print("[%s]%s" % (k , sub_abs.sum()))

                # # step: 计算unsat_node 重要度和 各个基学习器feature_importance的差值的绝对值之和
                # for dt, f_i in self._dts_importance.items():

                #     dt_abs =  abs(f_i/100 - node_cpr_data/100)
                #     # print("[%s]%s" % (k , sub_abs.sum()/100))
                #     print("tree_%s:%s" % (dt, dt_abs.sum()/100))
                
                # step:显示 unsat_node 和 feature_importance 的组合图
                # self.unsat_feature_dot_pic(feature_names, n_sorted_idx, f_sorted_idx, k)
                
                # step:单独显示 unsat_node 图
                # self.unsat_node_dot_pic(feature_names, n_sorted_idx, k)

                # break
    def clf_dt_precit(self, org_data, smt_data):
        i = 0
        for estimator in self._clf.estimators_:
            print('tree_%s'% i)
            org_r = estimator.predict(org_data)
            smt_r = estimator.predict(smt_data) 
            if org_r != smt_r:
                print('org[%s] smt[%s]' % (org_r, smt_r ))
                # print(estimator.)
            i+=1
                
    def get_clf_dt_importance(self):
        """
          显示模型中 各个基学习器的 feature_importance 特征
        """
        i = 0
        for estimator in self._clf.estimators_:
            # print('tree_%s'% i)
            feature_names = np.arange(784).reshape(784,)
            feature_importance = estimator.feature_importances_
            # make importances relative to max importance
            # feature_importance = 100.0 * (feature_importance / feature_importance.max())
            # f_sorted_idx = np.argsort(feature_importance)

            # plt.scatter(feature_names, f_sorted_idx, marker = '.',color = 'blue', s = 15, label = 'tree_%s_importance' % i )
            # plt.legend(loc = 'best')    # 设置 图例所在的位置 使用推荐位置
            # plt.savefig('tree_%s_feature_importance.jpg' % i, dpi=300) 
            # plt.clf()
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
        # for pix in range(255):
        # for pix in [255]:

        # test = [0]*784
        # for i in unsat_index_list:
        #     test[i] += 1
        pix = 1
        for i in range(data.shape[0]):
            if i not in unsat_index_list:
                tmp1[i] += 50
            else:
                tmp1[i] += pix

        # show_data1 = np.array(test).reshape(28, 28)
        # # plt.title('not in unsat:%s'%self._clf.predict(test.reshape(1, -1)))
        # plt.imshow(show_data1)
        # plt.show()
        # plt.clf()

        # if self._clf.predict(tmp1.reshape(1, -1))[0] !=0:
        #     print("[%s]not in unsat predict:%s"%(pix, self._clf.predict(tmp1.reshape(1, -1))[0]))
        # else:

        # break
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

    def output_test_data(self):
        """
        导出测试数据集
        """
        result_path = os.path.join(self._top_dir, 'result')

        # test_data= []
        for f in os.listdir(result_path):
            # print('-'*20)
            if f.endswith('.json'):
               
            # if f.endswith('.json') and f.startswith('9_verify'):
             # and f.startswith('0_verify'):

                # print('file:%s'% f)
                filePath = os.path.join(result_path, f)
                
                resultFile = open(filePath, 'r')
                result_dict = json.load(resultFile)

                org_class = result_dict[ORG_CLASS_KEY]
                org_data = result_dict[ORG_DATA_KEY]
                org_data.append(org_class)
                test_data.append(org_data)
        
        output = pd.DataFrame(test_data)
        output.to_csv("datasets/%s.csv" % self._model, encoding="utf-8-sig", header=False, index=False)


    def show_pic(self, num_row, num_clo, pic_list, pix):
        """
        sample: 这里输入的 sample 类型应该是(1, -1)
        plt image show 接受的是一个图片矩阵，而不是以个列表
        """

        for idx, pic_data in enumerate(pic_list):
            sample, pic_name = pic_data
            s_sample = sample.reshape(pix, pix)
            plt.subplot(num_row, num_clo, idx+1, facecolor='r')
            plt.title(pic_name)
            plt.figure(1, figsize=(3, 3))
            plt.imshow(s_sample, cmap=plt.cm.gray_r, interpolation='nearest')
            # plt.imshow(s_sample)
        plt.show()
        plt.clf()
        # plt.savefig(self._resultFile.replace('result', 'pic').replace('.json', '.jpg'))

    def analyse_verify_result(self):

        # feature_importance = self._clf.feature_importances_
        # print(feature_importance.shape)

        result_path = os.path.join(self._top_dir, 'result')

        proved_num = 0
        timeout_num = 0
        total_num = 0

        tmp_file = []
        unsat_tree_dict = {}
        unsat_node_dict = {}
        proved_dict = {}  # 记录该类别的 proved 的样本数
        timeout_dict = {}
        class_example_num = {} #记录该类别总验证样本数
        num_data = 784    

        base_class_data = {}

        # class_type = 9
        # for n in [1]:
        #     base_data = [0] * 784

        for f in os.listdir(result_path):
            # print('-'*20)
            if f.endswith('.json'):
               
            # if f.endswith('.json') and f.startswith('6_verify'):
             # and f.startswith('0_verify'):

                # print('file:%s'% f)
                filePath = os.path.join(result_path, f)
                
                resultFile = open(filePath, 'r')
                result_dict = json.load(resultFile)

                org_class = result_dict[ORG_CLASS_KEY]
                org_data = result_dict[ORG_DATA_KEY]

                # self.check_predict_result(org_data)
                org_predict = self._clf.predict(np.array(org_data).reshape(1, -1))[0]

                # print("org_predict:%s org_class:%s" % (org_predict, org_class))
                # 去除原模型识别不正确的样本
                if org_predict != org_class:
                    continue
                
                class_name = f.split('_')[0]  
                proved_dict.setdefault(class_name, 0)
                timeout_dict.setdefault(class_name, 0)
                class_example_num.setdefault(class_name, 0)
                base_class_data.setdefault(class_name, [0]*784)
  
                total_num += 1 
                class_example_num[class_name] +=1
                # break
                verify_result = result_dict[RESULT_KEY]

                # for i in range(784):
                #     # 如果该节点出现过的话，就+1
                #     if org_data[i] > 0:
                #         base_class_data[class_name][i] +=1

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

                    # 测试 unsat_list中的节点，对鲁棒性的影响
                    # self.check_node_pertubition(org_data, unsat_node_list)

                    sample = np.array(robust_data).reshape(28, 28)
                    
                    unsat_tree_dict.setdefault(class_name, [])
                    unsat_tree_dict[class_name].extend(unsat_tree_list)

                    unsat_node_dict.setdefault(class_name, [])
                    unsat_node_dict[class_name].extend(unsat_node_list)

                    proved_dict[class_name] += 1
                    
                    # org_predict = self._clf.predict(np.array(org_data).reshape(1, -1))[0]
                    # print("org:%s" % (org_predict))
                    # break

                    # unsat_pic= [0]*784
                    # unsat_index_list = [int(n.replace('x', '')) for n in unsat_node_list]
                    # for idx in unsat_index_list:
                    #     unsat_pic[idx] = 1

                    # pictures = []    
                    # show_data3 = np.array(unsat_pic).reshape(28, 28)
                

                    # show_data2 = np.array(org_data).reshape(28, 28)
               

                    # pictures.append((show_data2, ""))
                    # pictures.append((show_data3, ""))
                    # self.show_pic(1, 2, pictures, 28)


                elif FAILED == verify_result:
                    # print("Property: " + FAILED)
                    
                    org_add_smt_data = org_data[:]

                    tst_data = []
                    # smt_data = [0]*len(org_data)
                    smt_data = org_data[:]
                    wl_data = {}
                    for k, v in result_dict.items():
                        if k.startswith('x'):
                            # if smt_data[int(k.replace('x', ''))] == 0:
                            smt_data[int(k.replace('x', ''))] = int(v)
                            if int(v) != 0:
                                tst_data.append(int(v))    
                            org_add_smt_data[int(k.replace('x', ''))] = int(v)
                        elif k.startswith('wl_'):
                            index = k.split('_')[-1]
                            # print(index)
                            wl_data.setdefault(index, 0)
                            # wl_data[index] += int(v)
        
                    # 显示反例对比结果
                    # show_smt = np.array(smt_data).reshape(28, 28)
            
                    # show_org = np.array(org_data).reshape(28, 28)
            
                    # diff = [0]*784
                    # for idx in range(784):
                    #     if org_data[idx] != smt_data[idx]:
                    #         # print("org[{0}]:{1} smt:{2}".format(idx,org_data[idx],smt_data[idx]))
                    #         diff[idx]=abs(org_data[idx]-smt_data[idx])
                    # # print(diff)
                    # pictures = []
                    # pictures.append((show_org, 'Class: %s'%(FASHION_CLASS_DIC[org_class])))
                    # pictures.append((show_smt, 'Class: %s'%(FASHION_CLASS_DIC[self._clf.predict(show_smt.reshape(1,-1))[0]])))
                    # pictures.append((np.array(diff), 'Perturbation'))

                    # pictures.append((show_org, 'Class: %s' % self._clf.predict(show_org.reshape(1,-1))[0]))
                    # pictures.append((show_smt, 'Class: %s'%self._clf.predict(show_smt.reshape(1,-1))[0]))
                    # pictures.append((np.array(diff), 'Perturbation'))

                    # self.show_pic(1, 3, pictures, 28)

                    # 测试 base jpg 与 importance 的关系
                    # little_index_list = []

                    # for i in range(784):
                    #     if org_data[i] != smt_data[i] and smt_data[i] <=2:
                    #         # print("[%s]: org[%s] smt[%s]" % (i, org_data[i], smt_data[i]))
                    #         little_index_list.append(i)

                    # t = 0
                    # for estimator in self._clf.estimators_:
                        
                    #     fi_value = []
                    #     feature_names = np.arange(784).reshape(784,)
                    #     feature_importance = estimator.feature_importances_     
                    #     for idx in little_index_list:
                    #         # print('[%s]idx_fi:%s' % (idx, feature_importance[idx]))
                    #         fi_value.append(feature_importance[idx])
                    #     print('tree_%s:%s' % (t, np.mean(fi_value)))
                    #     t +=1
    
                    smt_predict = self._clf.predict(np.array(smt_data).reshape(1, -1))[0]

                    # self.clf_dt_precit(np.array(org_data).reshape(1, -1),np.array(smt_data).reshape(1, -1))
                    # print('smt_predict:[%s]'%smt_predict)
                elif TIMEOUT == verify_result:
                    timeout_dict[class_name] += 1
                    timeout_num +=1
                    
            # break
        # for k, v in unsat_tree_dict.items():
        #     tree = Counter(v)
        #     print('%s unsat tree' % k)
        #     print(tree)

        #     nodes = Counter(unsat_node_dict[k])
        #     print('%s unsat node' % k)
        #     print(nodes)
        # for k, v in base_class_data.items():
        #     self.calculate_base_and_mfi(k, v)

        # avg = np.mean(self.distance)
        # print("ave:%s" % avg)

        # 处理 各个类型的 unsat 特征
        # self.show_unsat_node(unsat_node_dict)

        # 统计不同类别的 鲁棒性
        for k, v in proved_dict.items():
            base = class_example_num[k]
            ratio = v/base

            timeout_num_item = timeout_dict[k]
            timeout_ratio = timeout_num_item/base
            print("[%s] total_num:%s robust ratio:%s timeout ratio:%s" %(k, base, ratio, timeout_ratio))

        # # 统计总体模型的鲁棒性
        print("total sample num:{}".format(total_num))
        print('proved ratio:{}'.format(proved_num /total_num))
        print('timeout ratio:{}'.format(timeout_num /total_num))


if __name__ == "__main__":
    # for model in ['rf_random_mnist_25_5_1_robust','rf_random_mnist_25_8_1_robust','rf_random_mnist_25_10_1_robust',
    #              'rf_random_mnist_50_5_1_robust','rf_random_mnist_50_8_1_robust','rf_random_mnist_50_10_1_robust',
    #              'rf_random_mnist_75_5_1_robust','rf_random_mnist_75_8_1_robust','rf_random_mnist_75_10_1_robust',
    #              'rf_random_mnist_100_5_1_robust','rf_random_mnist_100_8_1_robust','rf_random_mnist_100_10_1_robust']:
    # for model in ['rf_random_mnist_75_5_1_robust','rf_random_mnist_75_8_1_robust','rf_random_mnist_75_8_1_robust']:
    # for model in ['rf_random_mnist_25_5_3_robust','rf_random_mnist_25_8_3_robust','rf_random_mnist_25_10_3_robust','rf_random_mnist_50_8_3_robust','rf_random_mnist_50_5_3_robust']:
     # for model in ['rf_fashion_mnist_75_8_3_robust','rf_fashion_mnist_100_8_3_robust']:
     # for model in ['rf_fashion_mnist_20_3_1_robust']:
     for model in ['rf_random_mnist_25_5_3_robust','rf_random_mnist_25_8_3_robust','rf_random_mnist_25_10_3_robust',
                 'rf_random_mnist_50_5_3_robust','rf_random_mnist_50_8_3_robust','rf_random_mnist_50_10_3_robust',
                 'rf_random_mnist_75_5_3_robust','rf_random_mnist_75_8_3_robust','rf_random_mnist_75_10_3_robust',
                 'rf_random_mnist_100_5_3_robust','rf_random_mnist_100_8_3_robust','rf_random_mnist_100_10_3_robust']:
        print("%s --------" % model)
        ea1 = Experment_Analyses(model)
        ea1.load_model()
        # ea1.analyse_verify_result()
        # ea1.prepare_verify_time_data()
        ea1.calculate_verify_time()

        ea1.save_verify_time_result()
        # break

        # test_data = ea1.show_train_ratio('./datasets/mnist_1_test.csv')
        # show_dot_pic(clf1, clf2, test_data)
        # ea.show_train_ratio('rf_mnist_20_3_1_robust/model/rf_mnist_20_3_1_robust.csv')
    # ea.show_train_ratio('sklearn_rf/rf_mnist_20_8_1_robust_ib/model/rf_mnist_20_8_1_robust_ib.csv')
        # ea.show_train_ratio('./datasets/mnist_test.csv')

    # mnist = pd.read_csv('./datasets/mnist.csv')
