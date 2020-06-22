from numpy import genfromtxt
import numpy as np
import os

class HandleData(object):

    def __init__(self, oneHotFlag=False):
        self.total_data = genfromtxt('./Dround_Data_New/Nomalized/deg_0_normalize.csv', delimiter=',', dtype=np.float32).shape[0]*len(os.listdir('./Dround_Data_New/Nomalized'))
        self.data_per_angle = genfromtxt('./Dround_Data_New/Nomalized/deg_0_normalize.csv', delimiter=',', dtype=np.float32).shape[0]
        self.num_angles = len(os.listdir('./Dround_Data_New/Nomalized'))
        self.current_point = 0
        self.data_set = np.zeros((self.total_data, 4), dtype=np.float32)
        if oneHotFlag == True:
            self.label_set = np.zeros((self.total_data, self.num_angles), dtype=np.float32)
        else:
            self.label_set = [0 for i in range(self.total_data)]
        self.oneHotFlag = oneHotFlag

    def onehot_encode(self, number):
        encoded_no = np.zeros(self.num_angles, dtype=np.float32)
        if number < self.num_angles:
            encoded_no[number] = 1
        return encoded_no

    def get_synthatic_data(self):
        x_0 = genfromtxt('./Dround_Data_New/Nomalized/deg_0_normalize.csv', delimiter=',', dtype=np.float32)
        x_45 = genfromtxt('./Dround_Data_New/Nomalized/deg_45_normalize.csv',delimiter=',', dtype=np.float32)
        x_90 = genfromtxt('./Dround_Data_New/Nomalized/deg_90_normalize.csv', delimiter=',', dtype=np.float32)
        x_135 = genfromtxt('./Dround_Data_New/Nomalized/deg_135_normalize.csv',delimiter=',', dtype=np.float32)
        x_180 = genfromtxt('./Dround_Data_New/Nomalized/deg_180_normalize.csv', delimiter=',', dtype=np.float32)
        x_225 = genfromtxt('./Dround_Data_New/Nomalized/deg_225_normalize.csv',delimiter=',', dtype=np.float32)
        x_270 = genfromtxt('./Dround_Data_New/Nomalized/deg_270_normalize.csv', delimiter=',', dtype=np.float32)
        x_315 = genfromtxt('./Dround_Data_New/Nomalized/deg_315_normalize.csv',delimiter=',', dtype=np.float32)

        data_matrix = np.array([x_0,x_45, x_90,x_135, x_180,x_225, x_270,x_315], np.float32)

        for i in range(0, self.num_angles):
            for j in range(0, self.data_per_angle):
                "add one hot"
                "add data"
                if self.oneHotFlag == True:
                    self.label_set[i * self.data_per_angle + j] = self.onehot_encode(i)
                else:
                    self.label_set[i * self.data_per_angle + j] = i
                self.data_set[i * self.data_per_angle + j] = data_matrix[i][j]

        return self.data_set ,self.label_set