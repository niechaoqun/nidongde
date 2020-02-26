#encoding:utf8

import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
import pandas as pd


def convert_train_data(file_dir):
    root_dir = './datasets/gtsbr'

    directories = [file for file in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, file))]

    for files in directories:
        path = os.path.join(root_dir, files)
        if not os.path.exists(path):
            os.makedirs(path)

        data_dir = os.path.join(file_dir, files)

        file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".ppm")]
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                csv_dir = os.path.join(data_dir, f)   
        csv_data = pd.read_csv(csv_dir)

        csv_data_array = np.array(csv_data)
        for i in range(csv_data_array.shape[0]):
            csv_data_list = np.array(csv_data)[i,:].tolist()[0].split(";")
            sample_dir = os.path.join(data_dir, csv_data_list[0])
            img = PIL.Image.open(sample_dir)
            box = (int(csv_data_list[3]),int(csv_data_list[4]),int(csv_data_list[5]),int(csv_data_list[6]))
            roi_img = img.crop(box)
            new_dir = os.path.join(path, csv_data_list[0].split(".")[0] + ".jpg")
            roi_img.save(new_dir, 'JPEG')
            # break
        # break

def convert_train_data(file_dir):
    root_dir = './datasets/gtsbr'

    directories = [file for file in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, file))]

    for files in directories:
        path = os.path.join(root_dir, files)
        if not os.path.exists(path):
            os.makedirs(path)

        data_dir = os.path.join(file_dir, files)

        file_names = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".ppm")]
        for f in os.listdir(data_dir):
            if f.endswith(".csv"):
                csv_dir = os.path.join(data_dir, f)   
        csv_data = pd.read_csv(csv_dir)

        csv_data_array = np.array(csv_data)
        for i in range(csv_data_array.shape[0]):
            csv_data_list = np.array(csv_data)[i,:].tolist()[0].split(";")
            sample_dir = os.path.join(data_dir, csv_data_list[0])
            img = PIL.Image.open(sample_dir)
            box = (int(csv_data_list[3]),int(csv_data_list[4]),int(csv_data_list[5]),int(csv_data_list[6]))
            roi_img = img.crop(box)
            new_dir = os.path.join(path, csv_data_list[0].split(".")[0] + ".jpg")
            roi_img.save(new_dir, 'JPEG')
            
if __name__ == "__main__":
    convert_train_data('./datasets/GTSBR_SOURCE/Images')
