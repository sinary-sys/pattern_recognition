# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt

path = "F:\Mirror\学习资料\研一\pattern_recognition\第一次作业\作业数据_2021合成.xls"
data = pd.read_excel(path)
Vital_capacity = data['肺活量']
gender = data['性别 男1女0']
print(Vital_capacity)
print(gender)