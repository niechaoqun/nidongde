#encoding=utf-8

from z3 import *

Z3_MAX_FORMULAR_ITEM = 250

class RF_SMT_Z3(object):
    fileIndex = 0
    def __init__(self, isRegression=False, verify_dir=None, result_dir=None):
 
        """
        :param isRegression: 区别分类任务
        :param orginal_data: 测试样例的原数据:格式:[[1,2,3,4,5,]]
        :param epsilon:  约束特征值的扰动范围
        :param delta:    约束预测值的扰动范围
        """
        self._formulas = None
        self._isRegression = isRegression
        self._verify_dir = verify_dir
        self._result_dir = result_dir

        self._orginal_data = None
        self._orginal_class = None
        self._epsilon = None
        self._delta = None
        self._model_name = None
        self._smtFileName = None
        self._smt_result_file_name = None

        

    def set_test_sample_info(self, org_data=None, org_class=None, epsilon=None, delta=None, model_name=None):

        self._orginal_data = org_data[0]

        # print(self._orginal_data)
        self._orginal_class = org_class
        self._epsilon = epsilon
        self._delta = delta
        self._model_name = model_name
        self._smtFileName = '%s_verify_%s' % (self._orginal_class, RF_SMT_Z3.fileIndex)
        self._smt_result_file_name = None

    def get_smt_verify_pyFile_name(self):
        return '%s/%s.py' % (self._verify_dir, self._smtFileName)

    def get_smt_verify_result_file(self):
        return  self._smt_result_file_name   

    def save_formula_2_file(self, lines):
        with open('%s/%s.py' % (self._verify_dir, self._smtFileName), 'w') as f:
            f.writelines(line + '\n' for line in lines)
        RF_SMT_Z3.fileIndex += 1

    def devide_max_z3_formula(self, formulas, expr_name, formula_opr):
        """
        因为z3 对公式的最大 item 有限制为 253 项，假如公式大于该值的话，需要分割表达式项数
        :param formulas: 表达式项的列表
        :param expr_name: 表达式名字
        :param formula_opr: eg: 'Or','And' ...
        :return:
        """
        formulas_list = []
        formulas_name = []

        formulas_len = len(formulas)

        if formulas_len > Z3_MAX_FORMULAR_ITEM:

            index = 0
            start_split = 0
            for i in range(100, formulas_len, Z3_MAX_FORMULAR_ITEM):
                tmp_name = '%s%s'%(expr_name, index)
                formulas_list.append("%s=%s(%s)\n" % (tmp_name, formula_opr, ','.join(formulas[start_split:i])))
                formulas_name.append(tmp_name)
                start_split = i
                index += 1
            tmp_name = '%s%s' % (expr_name, index)
            formulas_list.append("%s=%s(%s)\n" % (tmp_name, formula_opr, ','.join(formulas[start_split:])))
            formulas_name.append(tmp_name)
        else:
            formulas_name.append(expr_name)
            formulas_list.append("%s=%s(%s)\n" % (expr_name, formula_opr, ','.join(formulas)))

        return formulas_list,formulas_name

    def solve_formula_counter(self, formulas, org_predict):
        """
           该函数用来构建反例测试的 z3 脚本
        """

        self._formulas = formulas

        file_lines = []

        regression_formulas = []
        total_feature = []

        r = 0 #记录回归器的 index
        regression_and_formulas = []
        regressor_and_names = {}

        value_list = [] #保存基学习器的预测值
        for tree_formulas, features in formulas:
            tree_dnf_formulas = [] #保存单颗树的 dnf 公式
            #开始构造z3 表达式
            regressor_and_names[r]=[]
            for i, f in enumerate(tree_formulas.values()):
                tree_dnf_formulas.append('And(%s)'% f)
                regression_and_formulas.append('and_%s_%s = And(%s)'%(r, i, f))
                regressor_and_names[r].append('and_%s_%s'%(r, i))
            total_feature.extend(features)
            regression_formulas.append(tree_dnf_formulas) # 保存整个回归器的 cnf 公式
            r+=1

        file_lines.append("""from z3 import *
import numpy as np
import json
import matplotlib.pyplot as plt\n""")

        file_lines.append('s = Solver()\n')

        # 添加 最小核缩减
        file_lines.append('s.set("sat.core.minimize",True)')

        total_feature = list(set(total_feature))
        pertubations_constrains = []

        # 添加特征值变量声明
        for var in total_feature:
            file_lines.append("x{0} = Int('x{0}')".format(var))
            # pertubations_constrains.append( 'x%s>=0' % var)
        
        # 添加 unsat 变量追踪值
        for var in total_feature:
            # 反例验证变量值
            file_lines.append( "s.assert_and_track({0}, '{1}')".format('x%s==%s' % (var, self._orginal_data[var]),'bx%s' % var))

        num_r = len(formulas)
        for i in range(0, num_r):
            file_lines.append("wl_%s=Int('wl_%s')" % (i, i))


        file_lines.append('M = AstMap()')
        for i in range(10):
            file_lines.append("r_%s=Int('r_%s')" % (i, i))
            file_lines.append("M[r_%s] = IntVal(0)"% i)

        file_lines.append("WL = [Int('wl_%s' % i) for i in range({0})]".format(num_r))
        file_lines.append("RL = [Int('r_%s' % i) for i in range(10)]")
        file_lines.append("for i in WL:")
        for i in range(10):
            file_lines.append("    M[r_{0}] = If(i == r_{0}, M[r_{0}] + 1, M[r_{0}])".format(i))
        
        file_lines.append("index = r_0")
        file_lines.append("out = M[r_0]")
        file_lines.append("for i in RL:")
        file_lines.append("    index = If(M[i] > out, i, index)")
        file_lines.append("    out = If(M[i] > out, M[i], out)")

        for f in regression_and_formulas:
            file_lines.append(f)

        for r, ands in regressor_and_names.items():
            file_lines.append('or_%s = Or(%s)'%(r, ','.join(ands)))

            #添加unsat树的追踪值
            file_lines.append("s.assert_and_track(or_{0}, 'tree_{0}')".format(r))

        # 根据任务类型的得出不同的公式
        if self._isRegression:
            # TODO: 回归任务的扰动值
            # 回归任务：各基学习器之和
            prediction = sum(value_list)
            for i in range(0, num_r):
                # print("wl_%s=Real('wl_%s')" % (i, i))
                file_lines.append("wl_%s=Real('wl_%s')" % (i, i))

            # print("out = %s" % '+'.join(['wl_%s' % i for i in range(0, num_r)]))
            file_lines.append("out = %s" % '+'.join(['wl_%s' % i for i in range(0, num_r)]))
            # print("robust = And(%s, (out-%s)>3)" % (','.join(['or_%s' % i for i in range(0, num_r)]),
            #                                         prediction))
            file_lines.append("robust = And(%s, (out-%s)>3)" % (','.join(['or_%s' % i for i in range(0, num_r)]),
                                                    prediction))
        else:
            solver_adds_names = []

            # 添加 整体模型公式
            robust_formulas = ['or_%s' % i for i in range(0, num_r)]
            rbt_formulas, rbt_formula_names = self.devide_max_z3_formula(robust_formulas, 'robust', 'And')
            for f in rbt_formulas:
                file_lines.append(f)
            solver_adds_names.extend(rbt_formula_names)

            # 添加预测值约束
            # 分类任务：基学习器中的最大值
            prediction = org_predict

            # out 值应该为
            solver_adds_names.append("index==%s"%(prediction))

            file_lines.append('s.add(And(%s))'% ','.join(solver_adds_names))

            file_lines.append('s.add(And(r_0==0, r_1==1, r_2==2, r_3==3, r_4==4, r_5==5, r_6==6, r_7==7, r_8==8, r_9==9))')


            org_data = self._orginal_data.reshape(-1).tolist()
    

            self._smt_result_file_name = '%s/%s.json' % (self._result_dir, self._smtFileName)
            file_lines.append('json_dict = {}')

            file_lines.append('print(s.check())')

            file_lines.append("json_dict['org_class']=%s" % self._orginal_class)
            file_lines.append("json_dict['org_data']=%s" % org_data)
            file_lines.append("if s.check() == unsat:")

            file_lines.append("    json_dict['result']='failed'")
            file_lines.append("    unsat_list = []")
            file_lines.append("    for i in s.unsat_core():")
            file_lines.append("        unsat_list.append('%s' % i)")
            file_lines.append("    json_dict['unsat'] = unsat_list")
            file_lines.append("else:")
            file_lines.append("    m = s.model()")
            file_lines.append("    json_dict['result']='proved'")
            file_lines.append("    for d in m.decls():")
            file_lines.append("        json_dict[d.name()] = '%s'%m[d]")
            file_lines.append("json_str = json.dumps(json_dict, sort_keys=True, indent=4)")
            file_lines.append("with open('%s', 'w') as f:" % self._smt_result_file_name)
            file_lines.append("    f.write(json_str)")

            self.save_formula_2_file(file_lines)

    def solve_formula_robust(self, formulas, org_predict):
        """
           该函数式用来构建 鲁棒性验证的脚本
        """

        self._formulas = formulas

        file_lines = []

        regression_formulas = []
        total_feature = []

        r = 0 #记录回归器的 index
        regression_and_formulas = []
        regressor_and_names = {}

        value_list = [] #保存基学习器的预测值
        for tree_formulas, features in formulas:
            tree_dnf_formulas = [] #保存单颗树的 dnf 公式
            #开始构造z3 表达式
            regressor_and_names[r]=[]
            for i, f in enumerate(tree_formulas.values()):
                tree_dnf_formulas.append('And(%s)'% f)
                regression_and_formulas.append('and_%s_%s = And(%s)'%(r, i, f))
                regressor_and_names[r].append('and_%s_%s'%(r, i))
            total_feature.extend(features)
            regression_formulas.append(tree_dnf_formulas) # 保存整个回归器的 cnf 公式
            r+=1

        file_lines.append("""from z3 import *
import numpy as np
import json
import matplotlib.pyplot as plt\n""")

        file_lines.append("def abs(a):")
        file_lines.append("    return If(a<0, -a, a)")

        file_lines.append('s = Solver()\n')

        # 添加 最小核缩减
        file_lines.append('s.set("sat.core.minimize",True)')

        #去除路径公式中的节点冗余
        total_feature = list(set(total_feature))
        pertubations_constrains = []

        # 添加特征值变量声明
        for var in total_feature:
            # print("x{0} = Int('x{0}')".format(var))
            file_lines.append("x{0} = Int('x{0}')".format(var))
            pertubations_constrains.append( 'x%s>=0' % var)
        
        # 添加 unsat 变量追踪值
        for var in total_feature:
            file_lines.append( "s.assert_and_track({0}, '{1}')".format('abs(x%s-%s)<=%s' % (var, self._orginal_data[var], self._epsilon),'bx%s'%var))
           
        # 添加结果判断依据,每个叶子节点都保存各个类别的权重值   
        num_r = len(formulas)
        for i in range(0, num_r):
            for idx in range(10):
                file_lines.append("wl_{0}_{1}=Real('wl_{0}_{1}')".format(i, idx))

        # 添加计算预测值的结果 map
        file_lines.append('M = AstMap()')
        for i in range(10):
            file_lines.append("r_%s=Int('r_%s')" % (i, i))
            file_lines.append("M[r_%s] = RealVal(0)"% i)

        # 添加各类型的结果声明, 10 代表的是类别个数：minist 的类别个数为10
        for idx in range(10):
            file_lines.append("WL_{0} = [Real('wl_%s_{0}' % i) for i in range({1})]".format(idx, num_r))

        file_lines.append("RL = [Int('r_%s' % i) for i in range(10)]")

        for idx in range(10):
            file_lines.append("for i in WL_{}:".format(idx))
            file_lines.append("    M[r_{0}] += i".format(idx))
        
        file_lines.append("index = r_0")
        file_lines.append("out = M[r_0]")
        file_lines.append("for i in RL:")
        file_lines.append("    index = If(M[i] > out, i, index)")
        file_lines.append("    out = If(M[i] > out, M[i], out)")

        for f in regression_and_formulas:
            file_lines.append(f)

        for r, ands in regressor_and_names.items():
            if len(ands) > Z3_MAX_FORMULAR_ITEM:
                or_formulas, or_div_names = self.devide_max_z3_formula(ands, 'or_%s_' % r, 'Or')
                for f in or_formulas:
                    file_lines.append(f)
                file_lines.append('or_%s=Or(%s)' % (r, ','.join(or_div_names)))

            else:
                file_lines.append('or_%s = Or(%s)'%(r, ','.join(ands)))

            # 添加对决策树的变量追踪
            file_lines.append("s.assert_and_track(or_{0}, 'tree_{0}')".format(r))

        # 特征值的扰动约束值的处理应该修改成绝对值的形式： real 类型
        # 特征值的扰动约束值的处理应该修改成集合的形式：int 类型
        # 根据任务类型的得出不同的公式
        if self._isRegression:
            # TODO: 回归任务的扰动值
            # 回归任务：各基学习器之和
            prediction = sum(value_list)
            for i in range(0, num_r):
                file_lines.append("wl_%s=Real('wl_%s')" % (i, i))

            file_lines.append("out = %s" % '+'.join(['wl_%s' % i for i in range(0, num_r)]))
            # print("robust = And(%s, (out-%s)>3)" % (','.join(['or_%s' % i for i in range(0, num_r)]),
            #                                         prediction))
            file_lines.append("robust = And(%s, (out-%s)>3)" % (','.join(['or_%s' % i for i in range(0, num_r)]),
                                                    prediction))
        else:
            solver_adds_names = []
            # 添加pertudations 约束公式
            pst_formulas, pst_formula_names = self.devide_max_z3_formula(pertubations_constrains, 'pertudations', 'And')
            for f in pst_formulas:
                file_lines.append(f)
            solver_adds_names.extend(pst_formula_names)

            # 添加 模型约束公式
            robust_formulas = ['or_%s' % i for i in range(0, num_r)]
            rbt_formulas, rbt_formula_names = self.devide_max_z3_formula(robust_formulas, 'robust', 'And')
            for f in rbt_formulas:
                file_lines.append(f)
            solver_adds_names.extend(rbt_formula_names)

            # 添加预测值约束
            # 分类任务：基学习器中的最大值
            prediction = org_predict

            # out 值应该为
            solver_adds_names.append("index!=%s"%(prediction))

            file_lines.append('s.add(And(%s))'% ','.join(solver_adds_names))

            file_lines.append('s.add(And(r_0==0, r_1==1, r_2==2, r_3==3, r_4==4, r_5==5, r_6==6, r_7==7, r_8==8, r_9==9))')


            org_data = self._orginal_data.reshape(-1).tolist()
    

            self._smt_result_file_name = '%s/%s.json' % (self._result_dir, self._smtFileName)
            file_lines.append('json_dict = {}')

            file_lines.append('print(s.check())')

            file_lines.append("json_dict['org_class']=%s" % self._orginal_class)
            file_lines.append("json_dict['org_data']=%s" % org_data)
            file_lines.append("if s.check() == unsat:")
            file_lines.append("    unsat_list = []")
            file_lines.append("    for i in s.unsat_core():")
            file_lines.append("        unsat_list.append('%s' % i)")
            file_lines.append("    json_dict['unsat'] = unsat_list")
            file_lines.append("    json_dict['result']='proved'")
            file_lines.append("else:")
            file_lines.append("    m = s.model()")
            file_lines.append("    json_dict['result']='failed'")
            file_lines.append("    for d in m.decls():")
            file_lines.append("        json_dict[d.name()] = '%s'%m[d]")
            file_lines.append("json_str = json.dumps(json_dict, sort_keys=True, indent=4)")
            file_lines.append("with open('%s', 'w') as f:" % self._smt_result_file_name)
            file_lines.append("    f.write(json_str)")

            self.save_formula_2_file(file_lines)

    