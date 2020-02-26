#encoding=utf-8

import numpy as np

SIGN_LEFT = "<="
SIGN_RIGHT = ">"

class RF_Tree_Extractor(object):
    def __init__(self, class_info=None, epsilon=None, estimators=None, is_rf=False):
        if is_rf:
            self._estimators = estimators
            self._class_info = class_info
        else:
            self._estimators = estimators.reshape(-1)
        
        self._add_reduce = False  #是否需要增加路径缩减；如果是为了验证获取鲁棒性的 unsat 则不需要添加
        self._sample = None
        self._is_rf = is_rf
        self._epsilon = epsilon

    def set_sample(self, sample):
        self._sample = sample

    def set_is_add_reduce(self, is_reduce):
        self._add_reduce = is_reduce

    def get_abs_min_max(self, data):
        min_num = data-self._epsilon
        max_num = data+self._epsilon

        if min_num <=0:
            min_num = 0

        return min_num, max_num

    def extract_formula(self, index, estimator, original_data):
        """

        :param index:
        :param estimator:
        :return: tree_formulas: 返回字典：key：叶子节点值 value：叶子节点对应的路径公式
                 tree_features: 返回 list： 保存用到的所有的 feature_id
        """
        #单颗书的公式集
        tree_formulas = {}
        leaves = list()
        tree_features = [] #保存 用到的 feature id

        left_dict = {}
        right_dict = {}

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        value = estimator.tree_.value

        # print("[NCQ]feature:%s" % feature)

        weighted_n_node_samples = estimator.tree_.weighted_n_node_samples
        threshold = estimator.tree_.threshold
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.

        # 树结构可以通过遍历的方式获得不同的特征：如 树的深度 和 判断节点是否为叶子节点
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth

        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True

            for i in range(n_nodes):
                if is_leaves[i] and i not in leaves:
                    leaves.append(i)
                else:
                    # feature_id = feature[i]
                    # if feature_id >=0 and feature_id not in tree_features:
                    #     tree_features.append(feature_id)
                    left_dict[children_left[i]] = i
                    right_dict[children_right[i]] = i

        decision_leaf_id = estimator.apply(self._sample)[0]
        for leaf in leaves:

            #根据 leaf 来获取 pai_path
            path = []
            sign_dict = {}
            self._get_path_formula(path, sign_dict, leaf, left_dict, right_dict)
            path.reverse()

            # 如果需要缩减路径的话，则根据节点的值进行 epsilon 的判断
            if self._add_reduce:
                is_path_conform = self.judge_path_conform_pertubations(path, sign_dict, feature, value, threshold, original_data)
            else:
                is_path_conform = True

            # 如果 符合 epsilon 的约束，则加入到树路径中去，默认都是添加的
            if is_path_conform:
                formula, path_features= self._formula_cst(index, path, sign_dict, feature, value, threshold)
                tree_formulas[leaf]=(",").join(formula)
                tree_features.extend(path_features)
        return tree_formulas, tree_features

    def is_need_reduce_unconform_path():
        return 


    def judge_path_conform_pertubations(self, path, sign_dict, feature, value, threshold, original_data):
        for i in path[:-1]:
            feature_name = feature[i]
            sign = sign_dict[i]
            threshold_value = threshold[i]
            test_value = original_data[feature_name]
            min_value, max_value = self.get_abs_min_max(test_value)

            if sign == SIGN_LEFT:
                if min_value <= threshold_value:
                    continue
                else:
                    return False
            else:
                if max_value > threshold_value:
                    continue
                else:
                    return False

        return True

    
    def _formula_cst(self, estimator_index, path, sign_dict, feature, value, threshold):
        formula = []
        path_features = []
        for i in path[:-1]:

            feature_id = feature[i]
            sign = sign_dict[i]
            threshold_value = threshold[i]
            
            # 添加中间节点约束
            formula.append("(x%s %s %s)" %(feature_id, sign, threshold_value))
            path_features.append(feature_id)

        # 添加叶子节点的约束

        # 如果模型为随机森林，叶子节点的值应该变为对应的类别
        if self._is_rf:
            # leaf_value = self._class_info[np.argmax(value[path[-1]][0])]
            leaf_value = value[path[-1]][0]
            tmp = np.array(leaf_value)
            leaf_prob = tmp/tmp.sum()
            for idx, v in enumerate(leaf_prob.tolist()):
                formula.append("(wl_%s_%s == %s)"% (estimator_index, idx, v))    
                # print("(wl_%s_%s == %s)"% (estimator_index, idx, v))
        else:
            leaf_value = value[path[-1]][0][0]
            formula.append("(wl_%s == %s)"% (estimator_index, leaf_value))
        return formula, path_features


    def _get_path_formula(self, path, sign_dict, node, child_left_back, child_right_back):
        path.append(node)
        if node ==0:
            return

        if node in child_left_back:
            node = child_left_back[node]
            sign_dict[node] = SIGN_LEFT
            self._get_path_formula(path, sign_dict, node, child_left_back, child_right_back)
        elif node in child_right_back:
            node = child_right_back[node]
            sign_dict[node] = SIGN_RIGHT
            self._get_path_formula(path, sign_dict, node, child_left_back, child_right_back)
        else:
            print("node not found!")

    def get_decision_tree_formulas(self, original_data):
        """
        :return: list(tuple)
                 t1: decision_leaf_id
                 t2: tree_formulas
        """
        regression_formulas = []
        for index, estimator in enumerate(self._estimators):
            tree_formula, tree_features = self.extract_formula(index, estimator, original_data[0])

            regression_formulas.append((tree_formula, tree_features))
        return regression_formulas

    def show(self):
        index = 0
        for estimator in self._estimators:
            # break
            print('-'*10)
            print('Tree:%s' % index)
            index+=1
            n_nodes = estimator.tree_.node_count
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            feature = estimator.tree_.feature

            value = estimator.tree_.value

            # 描述的是各个节点的加权样本数,也就是每个节点存在的样本个数
            weighted_n_node_samples = estimator.tree_.weighted_n_node_samples
            # print(weighted_n_node_samples)

            threshold = estimator.tree_.threshold

            # The tree structure can be traversed to compute various properties such
            # as the depth of each node and whether or not it is a leaf.

            # 树结构可以通过遍历的方式获得不同的特征：如 树的深度 和 判断节点是否为叶子节点

            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth

            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1

                # If we have a test node
                if (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True

            # 开始遍历显示树的结构
            print("The binary tree structure has %s nodes and has "
                  "the following tree structure:"
                  % n_nodes)
            for i in range(n_nodes):
                if is_leaves[i]:
                    # print("%snode=%s leaf node. value=%s" % (node_depth[i] * "\t", i, self._class_info[np.argmax(value[i][0])]))
                    print("%snode=%s leaf node. value=%s" % (node_depth[i] * "\t", i, value[i][0]))
                else:
                    print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                          "node %s."
                          % (node_depth[i] * "\t",
                             i,
                             children_left[i],
                             feature[i],
                             threshold[i],
                             children_right[i],
                             ))

