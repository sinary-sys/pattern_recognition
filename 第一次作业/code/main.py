# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt

# import numpy as np


path = "F:\\Mirror\\学习资料\\研一\\pattern_recognition\\第一次作业\\作业数据_2021合成.xls"
data = pd.read_excel(path)
man_Vital_capacity = data[data['性别 男1女0'].values == 1]
man_Vital_capacity = man_Vital_capacity['肺活量']
faman_Vital_capacity = data[data['性别 男1女0'].values == 0]
faman_Vital_capacity = faman_Vital_capacity['肺活量']

the_man_Vital_capacity = man_Vital_capacity.value_counts(dropna=False, normalize=True, sort=False)
the_faman_Vital_capacity = faman_Vital_capacity.value_counts(dropna=False, normalize=True)

# 正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = [16, 6]
# 使用样式
plt.style.use("ggplot")

# 指定分组个数
n_bins = 20
fig, ax = plt.subplots(figsize=(8, 5))
# 实际绘图代码与单类型直方图差异不大，只是增加了一个图例项
# 在 ax.hist 函数中先指定图例 label 名称
ax.hist([man_Vital_capacity.values, faman_Vital_capacity.values], n_bins, histtype='bar', label=list('男女'))

ax.set_title('男女生肺活量统计直方图')

# 通过 ax.legend 函数来添加图例
ax.legend()
plt.show()

fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=80)
width_1 = 50
ax1.bar(the_man_Vital_capacity.index, the_man_Vital_capacity.values, width=width_1, label="男生")
ax1.bar(the_faman_Vital_capacity.index + width_1, the_faman_Vital_capacity.values, width=width_1, label="女生")

ax1.legend()
plt.show()
