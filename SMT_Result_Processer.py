#encoding=utf-8

import json
import numpy as np
import matplotlib.pyplot as plt


ORG_CLASS_KEY = 'org_class'
ORG_DATA_KEY = 'org_data'
RESULT_KEY = 'result'
PROVED = 'proved'
FAILED = 'failed'

class Verify_Result_Processer(object):
    """
    处理 验证结果，构建生成的反例样本
    """
    def __init__(self, pic_dir):
        self._resultFile = None
        self._pic_dir = pic_dir
        self._pix = None

    def set_info(self, resultFile, pix):
        self._resultFile = resultFile
        self._pix = pix

    def show_pic(self, num_row,num_clo, pic_list):
        """
    	sample: 这里输入的 sample 类型应该是(1, -1)
    	plt image show 接受的是一个图片矩阵，而不是以个列表
    	"""

        for idx, pic_data in enumerate(pic_list):
            sample, pic_name = pic_data
            s_sample = sample.reshape(self._pix, self._pix)
            plt.subplot(num_row, num_clo, idx+1, facecolor='r')
            plt.title(pic_name)
            plt.figure(1, figsize=(3, 3))
            # plt.imshow(s_sample, cmap=plt.cm.gray_r, interpolation='nearest')
            plt.imshow(s_sample)
        plt.savefig(self._resultFile.replace('result', 'pic').replace('.json', '.jpg'))
        # plt.show()

    def parse_counter_resultFile(self, model_predict):
        """
          处理countereexample 的 result 文件
        """

        resultFile = open(self._resultFile, 'r')
        result_dict = json.load(resultFile)
        verify_result = result_dict[RESULT_KEY]
        if FAILED == verify_result:
            print("Property: " + FAILED)

            org_class = result_dict[ORG_CLASS_KEY]
            org_data = result_dict[ORG_DATA_KEY]

            unsat_list = result_dict['unsat']

            robust_data = [0] * len(org_data)
            for pix in unsat_list:
                if pix.startswith('bx'):
                    robust_data[int(pix.replace('bx', ''))] = org_data[int(pix.replace('bx', ''))]
            
            pictures = []

            pictures.append((np.array(org_data), 'original'))
            pictures.append((np.array(robust_data), 'unsat_pic:(%s)' % model_predict))

            self.show_pic(1, 2, pictures)


    def parse_robust_resultFile(self, model_predict):
        """
          处理robust 的 result 文件
        """
        resultFile = open(self._resultFile, 'r')
        result_dict = json.load(resultFile)
        verify_result = result_dict[RESULT_KEY]

        org_class = result_dict[ORG_CLASS_KEY]
        org_data = result_dict[ORG_DATA_KEY]

        pictures = []
        
        pictures.append((np.array(org_data), 'original(model_predict:%s)'% model_predict))
        if PROVED == verify_result:
            print("Property: " + PROVED)
            unsat_list = result_dict['unsat']
            robust_data = [0] * len(org_data)
            for pix in unsat_list:
                if pix.startswith('bx'):
                    robust_data[int(pix.replace('bx', ''))] = org_data[int(pix.replace('bx', ''))]
            pictures.append((np.array(robust_data), 'unsat_pic'))

        elif FAILED == verify_result:

            print("Property: " + FAILED)
            smt_data = [0]*len(org_data)
            wl_data = {}
            for k, v in result_dict.items():
                if k.startswith('x'):
                    smt_data[int(k.replace('x', ''))] = int(v)
                elif k.startswith('wl_'):
                    wl_data[k.replace('wl_', '')] = eval(v+'.0')
            pictures.append((np.array(smt_data), 'counter_pic'))

        self.show_pic(1, 2, pictures)


if __name__ == "__main__":

    vr = Verify_Result_Processer('.')
    vr.set_info("rf_mnist_20_8_1/result/0_verify_0.json", 28)
    vr.parse_smt_resultFile('test.jpg')




