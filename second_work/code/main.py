import pandas as pd
import sklearn.datasets
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plt
import numpy as np

k = 2

path = '../作业数据_2021合成.xls'
data = pd.read_excel(path)  # 读入的数据结构为dataframe类型

man_x_h_w_50m_f = data[data['性别 男1女0'].values == 1]
faman_x_h_w_50m_f = data[data['性别 男1女0'].values == 0]

man_x_h_w = man_x_h_w_50m_f[['身高(cm)', '体重(kg)']].values
faman_x_h_w = faman_x_h_w_50m_f[['身高(cm)', '体重(kg)']].values

p1 = plt.scatter(man_x_h_w[:,0], man_x_h_w[:,1], c='g', marker='*', linewidths=1)
p2 = plt.scatter(faman_x_h_w[:,0], faman_x_h_w[:,1], c='r', marker='*', linewidths=1)

plt.xlabel('height/cm')
plt.ylabel('weight/kg')
gender_label = ['boy', 'girl']
plt.legend([p1, p2], gender_label, loc=0)
plt.show()

x_h_w_50m_f = data[['身高(cm)', '体重(kg)', '50米成绩', '肺活量']].values
x_h_w = data[['身高(cm)', '体重(kg)']].values
