from cmath import nan, isnan

import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from constant import EXCEL_FILE, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, MAX_VALUE

class ExcelLoader:
    """
        Loading data from EXCEL_FILE
    """
    def __init__(self):
        """
            Reading data
        """
        self.work_space = pd.ExcelFile(EXCEL_FILE)
        self.data = []
        for station in self.work_space.sheet_names[1:]:
            _data = pd.read_excel(self.work_space, sheet_name=station)
            current_point_data = []
            for i in range(3, 31):
                current_point_data.extend(_data.values[i][31:43])
            self.data.append(current_point_data)
        
        self.input_data = []
        self.output_data = []
        for statistics in self.data:
            for i in range(len(statistics) - INPUT_SIZE - OUTPUT_SIZE):
                ok = True
                for e in statistics[i : i + INPUT_SIZE + OUTPUT_SIZE]:
                    if isnan(e):
                        ok = False
                if ok:
                    self.input_data.append(statistics[i : i + INPUT_SIZE])
                    self.output_data.append(statistics[i + INPUT_SIZE : i + INPUT_SIZE + OUTPUT_SIZE])
        
        print(len(self.input_data))
             
class DataProcessor:
    def __init__(self):
        self.o = ExcelLoader()
        
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.o.input_data, self.o.output_data, train_size = 0.9, test_size = 0.1, random_state = 0)




if __name__ == "__main__":
    o = DataProcessor()
    model = LinearRegression().fit(o.X_train, o.y_train)
    for i in range(20):
        print(model.predict([o.X_valid[i]]), o.y_valid[i])   