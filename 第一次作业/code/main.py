# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import numpy as np

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

man_norm = st.norm.fit(man_Vital_capacity.values)
faman_norm = st.norm.fit(faman_Vital_capacity.values)
print(man_norm, faman_norm)

man_norm_avg = sum(man_Vital_capacity.values) / len(man_Vital_capacity.values)
faman_norm_avg = sum(faman_Vital_capacity.values) / len(faman_Vital_capacity.values)
print('男生样本均值', man_norm_avg)
print('女生样本均值', faman_norm_avg)


# 正态分布的概率密度函数
def normpdf(x, mu, sigma):
    pdf = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


mu, sigma = (4300.950530035336, 766.7614177550078)
x = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.01)  # 生成数据，步长越小，曲线越平滑
y = normpdf(x, mu, sigma)

mu1, sigma1 = (3247.794117647059, 760.1970440930107)
x1 = np.arange(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 0.01)  # 生成数据，步长越小，曲线越平滑
y1 = normpdf(x1, mu1, sigma1)
# 概率分布曲线
plt.plot(x, y, 'r--', linewidth=2, label="man")
plt.plot(x1, y1, 'g--', linewidth=2, label="faman")
plt.title('Normal Distribution: mu = {:.2f}, sigma={:.2f}'.format(mu, sigma))

plt.vlines(mu, 0, normpdf(mu, mu, sigma), colors="c", linestyles="dotted")
plt.vlines(mu + sigma, 0, normpdf(mu + sigma, mu, sigma), colors="y", linestyles="dotted")
plt.vlines(mu - sigma, 0, normpdf(mu - sigma, mu, sigma), colors="y", linestyles="dotted")
plt.xticks([mu - sigma, mu, mu + sigma], ['μ-σ', 'μ', 'μ+σ'])

# 显示图例
plt.legend()
# 输出

plt.show()

man_height = data[data['性别 男1女0'].values == 1]
man_weight = man_height['体重(kg)']
man_height = man_height['身高(cm)']
faman_height = data[data['性别 男1女0'].values == 0]
faman_weight = faman_height['体重(kg)']
faman_height = faman_height['身高(cm)']

p_c1 = len(man_weight.values) / (len(man_weight.values) + len(faman_weight.values))
p_c2 = len(faman_weight.values) / (len(man_weight.values) + len(faman_weight.values))
print(p_c1, p_c2)


# 求协方差矩阵
def get_covariance_matrix_coefficient(arr1, arr2):  # arr1与arr2长度相等
    datalength1 = len(arr1)
    datalength2 = len(arr2)
    sum_temp = []
    for i in range(datalength1):
        sum_temp.append((arr1[i] - sum(arr1) / datalength1) * (arr2[i] - sum(arr2) / datalength2))
        c12 = sum(sum_temp)
    covariance_matrix_c12 = c12 / (datalength1 - 1)
    return covariance_matrix_c12


man_height_mean, man_height_std = st.norm.fit(man_height.values)  # 男生升高分布参数
man_weight_mean, man_weight_std = st.norm.fit(man_weight.values)  # 男生体重分布参数
woman_height_mean, woman_height_std = st.norm.fit(faman_height.values)  # 女生升高分布参数
woman_weight_mean, woman_weight_std = st.norm.fit(faman_weight.values)  # 女生体重分布参数
print('男生身高', man_height_mean, man_height_std)
print('男生体重', man_weight_mean, man_weight_std)
print('女生身高', woman_height_mean, woman_height_std)
print('女生体重', woman_weight_mean, woman_weight_std)
man_c11 = man_height_std ** 2
man_c22 = man_weight_std ** 2
man_c12 = man_c21 = get_covariance_matrix_coefficient(man_height.values, man_weight.values)
man_covariance_matrix = np.matrix([[man_c11, man_c12], [man_c21, man_c22]])
woman_c11 = woman_height_std ** 2
woman_c22 = woman_weight_std ** 2
woman_c12 = woman_c21 = get_covariance_matrix_coefficient(faman_height.values, faman_weight.values)
woman_covariance_matrix = np.matrix([[woman_c11, woman_c12], [woman_c21, woman_c22]])
print(man_covariance_matrix, woman_covariance_matrix)

man_feature_mean_vector = np.matrix([[man_height_mean], [man_weight_mean]])
woman_feature_mean_vector = np.matrix([[woman_height_mean], [woman_weight_mean]])


# 定义等高线高度函数
def f(sample_height, sample_weight):
    mytemp1 = np.zeros(shape=(100, 100))
    for i in range(100):
        for j in range(100):
            sample_vector = np.matrix([[sample_height[i, j]], [sample_weight[i, j]]])
            sample_vector_T = np.transpose(sample_vector)
            # 定义决策函数
            mytemp1[i, j] = 0.5 * np.transpose(sample_vector - man_feature_mean_vector) * (
                np.linalg.inv(man_covariance_matrix)) * \
                            (sample_vector - man_feature_mean_vector) - 0.5 * np.transpose(
                sample_vector - woman_feature_mean_vector) * \
                            (np.linalg.inv(woman_covariance_matrix)) * (sample_vector - woman_feature_mean_vector) + \
                            0.5 * math.log(
                (np.linalg.det(man_covariance_matrix)) / (np.linalg.det(woman_covariance_matrix))) - \
                            math.log(p_c1 / p_c2)
    return mytemp1


sample_height = np.linspace(150, 180, 100)
sample_weight = np.linspace(40, 80, 100)
# 将原始数据变成网格数据
Sample_height, Sample_weight = np.meshgrid(sample_height, sample_weight)
# 填充颜色
plt.contourf(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, alpha=0)
# 绘制等高线,圈内为女生，圈外为男生
C = plt.contour(Sample_height, Sample_weight, f(Sample_height, Sample_weight), 0, colors='black', linewidths=0.6)
# 显示各等高线的数据标签
plt.clabel(C, inline=True, fontsize=10)

# 显示男女生样本散点图

p1 = plt.scatter(man_height.values, man_weight.values, c='g', marker='+', linewidths=0.4)
p2 = plt.scatter(faman_height.values, faman_weight.values, c='r', marker='*', linewidths=0.4)


# 定义显示坐标函数
def Display_coordinates(m, n):
    plt.scatter(m, n, marker='s', linewidths=0.4)
    plt.annotate((m, n), xy=(m, n))
    return


# 并判断某样本的身高体重分别为(160,45)时应该属于男生还是女生？为(178,70)时呢
Display_coordinates(160, 45)
Display_coordinates(178, 70)
label = ['boy', 'girl']
plt.legend([p1, p2], label, loc=0)
plt.xlabel('height/cm')
plt.ylabel('weight/kg')
plt.show()


def get_mean_bayes(arr, mean0, variance0, variance):
    datasum = sum(arr)
    datalen = len(arr)
    mean_bayes = (variance0 * datasum + variance * mean0) / (datalen * variance0 + variance)
    return mean_bayes


print(get_mean_bayes(man_Vital_capacity.values, 3840, 562, 700))
print(get_mean_bayes(faman_Vital_capacity.values, 2661, 536, 700))