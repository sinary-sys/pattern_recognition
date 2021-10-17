import pandas as pd
import sklearn.datasets
import sklearn.cluster
import scipy.cluster.vq
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.integrate import quad
import math as m
from mpl_toolkits.mplot3d import Axes3D

k = 2

path = '../作业数据_2021合成.xls'
data = pd.read_excel(path)  # 读入的数据结构为dataframe类型

man_x_h_w_50m_f = data[data['性别 男1女0'].values == 1]
faman_x_h_w_50m_f = data[data['性别 男1女0'].values == 0]

man_x_h_w = man_x_h_w_50m_f[['身高(cm)', '体重(kg)']].values
faman_x_h_w = faman_x_h_w_50m_f[['身高(cm)', '体重(kg)']].values

p1 = plt.scatter(man_x_h_w[:, 0], man_x_h_w[:, 1], c='g', marker='*', linewidths=1)
p2 = plt.scatter(faman_x_h_w[:, 0], faman_x_h_w[:, 1], c='r', marker='*', linewidths=1)

plt.xlabel('height/cm')
plt.ylabel('weight/kg')
gender_label = ['boy', 'girl']
plt.legend([p1, p2], gender_label, loc=0)
plt.show()

x_h_w_50m_f = data[['身高(cm)', '体重(kg)', '50米成绩', '肺活量']].values
x_h_w = data[['身高(cm)', '体重(kg)']].values

man_x_h_w_50m_f = man_x_h_w_50m_f[['身高(cm)', '体重(kg)', '50米成绩', '肺活量']].values
faman_x_h_w_50m_f = faman_x_h_w_50m_f[['身高(cm)', '体重(kg)', '50米成绩', '肺活量']].values

# 聚类数为2
kmeans_n_clusters = 2
# 构造聚类器,聚类数为2
# kmeans_clusterer = KMeans(kmeans_n_clusters)
kmeans_clusterer = KMeans(n_clusters=kmeans_n_clusters, init='random', n_init=1)
# 训练聚类
kmeans_clusterer.fit(x_h_w)

kmeans_cluster_label = kmeans_clusterer.labels_  # 获取聚类标签
# print('kmeans_label_pred:',kmeans_cluster_label)
kmeans_cluster_center = kmeans_clusterer.cluster_centers_  # 获取聚类中心
# print('kmeans_cluster_center:',kmeans_cluster_center)
kmeans_inertia = kmeans_clusterer.inertia_  # 获取聚类准则的总和
# print('kmeans_inertia:',kmeans_inertia)

# 绘图，显示聚类结果
plt.figure(2)
plt.clf()
# markers = ['^', 'x', 'o', '*', '+']
colours = ['g', 'r', 'b', 'y', 'c']
# class_label   = ['Class 1','Class 2','Class 3','Class 4','Class 5','center']
class_label = ['Class 1', 'Class 2', 'Class 3', 'center']
for i in range(kmeans_n_clusters):
    kmeans_members = kmeans_cluster_label == i
    plt.scatter(x_h_w[kmeans_members, 0], x_h_w[kmeans_members, 1], s=30, c=colours[i], marker='.')
# plt.legend(class_label,loc=0)
plt.title('KMeans clustering result')
plt.xlabel('height/cm')
plt.ylabel('weight/kg')
plt.show()


def k_means(data, k, number_of_iterations):
    n = len(data)
    number_of_features = data.shape[1]
    # Pick random indices for the initial centroids.
    initial_indices = np.random.choice(range(n), k)
    # We keep the centroids as |features| x k matrix.
    means = data[initial_indices].T
    # To avoid loops, we repeat the data k times depthwise and compute the
    # distance from each point to each centroid in one step in a
    # n x |features| x k tensor.
    repeated_data = np.stack([data] * k, axis=-1)
    all_rows = np.arange(n)
    zero = np.zeros([1, 1, 2])
    for _ in range(number_of_iterations):
        # Broadcast means across the repeated data matrix, gives us a
        # n x k matrix of distances.
        distances = np.sum(np.square(repeated_data - means), axis=1)
        # Find the index of the smallest distance (closest cluster) for each
        # point.
        assignment = np.argmin(distances, axis=-1)
        # Again to avoid a loop, we'll create a sparse matrix with k slots for
        # each point and fill exactly the one slot that the point was assigned
        # to. Then we reduce across all points to give us the sum of points for
        # each cluster.
        sparse = np.zeros([n, k, number_of_features])
        sparse[all_rows, assignment] = data
        # To compute the correct mean, we need to know how many points are
        # assigned to each cluster (without a loop).
        counts = (sparse != zero).sum(axis=0)
        # Compute new assignments.
        means = sparse.sum(axis=0).T / counts.clip(min=1).T
    return means.T


fig = plt.figure(3)  # 创建一个图
plt.rcParams['font.sans-serif'] = ['SimHei']
ax = fig.add_subplot(111, projection='3d')

X, Y, Z = man_x_h_w_50m_f[:, 0], man_x_h_w_50m_f[:, 1], man_x_h_w_50m_f[:, 2]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
x1, y1, z1 = faman_x_h_w_50m_f[:, 0], faman_x_h_w_50m_f[:, 1], faman_x_h_w_50m_f[:, 2]

cm = plt.cm.get_cmap('jet')  # 颜色映射，为jet型映射规则

fig = ax.scatter3D(X, Y, Z, c=man_x_h_w_50m_f[:, 3], cmap=cm, marker='*')
fig = ax.scatter3D(x1, y1, z1, c=faman_x_h_w_50m_f[:, 3], cmap=cm, marker='.')

cb = plt.colorbar(fig)  # 设置坐标轴
ax.set_xlabel('身高')
ax.set_ylabel('体重')
ax.set_zlabel('50m')
cb.ax.tick_params(labelsize=12)
cb.set_label('肺活量', size=15)

plt.show()

# 聚类数为2
kmeans_n_clusters = 2
# 构造聚类器,聚类数为2
# kmeans_clusterer = KMeans(kmeans_n_clusters)
kmeans_clusterer = KMeans(n_clusters=kmeans_n_clusters, init='random', n_init=1)
# 训练聚类
kmeans_clusterer.fit(x_h_w_50m_f)

kmeans_cluster_label = kmeans_clusterer.labels_  # 获取聚类标签
# print('kmeans_label_pred:',kmeans_cluster_label)
kmeans_cluster_center = kmeans_clusterer.cluster_centers_  # 获取聚类中心
# print('kmeans_cluster_center:',kmeans_cluster_center)
kmeans_inertia = kmeans_clusterer.inertia_  # 获取聚类准则的总和
# print('kmeans_inertia:',kmeans_inertia)
fig1 = plt.figure(4)  # 创建一个图
plt.rcParams['font.sans-serif'] = ['SimHei']
ax1 = fig1.add_subplot(111, projection='3d')

kmeans_members = kmeans_cluster_label == 0
s0 = x_h_w_50m_f[kmeans_members]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
kmeans_members = kmeans_cluster_label == 1
s1 = x_h_w_50m_f[kmeans_members]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y

cm1 = plt.cm.get_cmap('jet')  # 颜色映射，为jet型映射规则

fig1 = ax1.scatter3D(s0[:, 0], s0[:, 1], s0[:, 2], c=s0[:, 3], cmap=cm1, marker='*')
fig1 = ax1.scatter3D(s1[:, 0], s1[:, 1], s1[:, 2], c=s1[:, 3], cmap=cm1, marker='.')

cb1 = plt.colorbar(fig1)  # 设置坐标轴
ax1.set_xlabel('身高')
ax1.set_ylabel('体重')
ax1.set_zlabel('50m')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('肺活量', size=15)

plt.show()

# 聚类数为2
kmeans_n_clusters = 3
# 构造聚类器,聚类数为2
# kmeans_clusterer = KMeans(kmeans_n_clusters)
kmeans_clusterer = KMeans(n_clusters=kmeans_n_clusters, init='random', n_init=1)
# 训练聚类
kmeans_clusterer.fit(x_h_w_50m_f)

kmeans_cluster_label = kmeans_clusterer.labels_  # 获取聚类标签
# print('kmeans_label_pred:',kmeans_cluster_label)
kmeans_cluster_center = kmeans_clusterer.cluster_centers_  # 获取聚类中心
# print('kmeans_cluster_center:',kmeans_cluster_center)
kmeans_inertia = kmeans_clusterer.inertia_  # 获取聚类准则的总和
# print('kmeans_inertia:',kmeans_inertia)
fig1 = plt.figure(5)  # 创建一个图
plt.rcParams['font.sans-serif'] = ['SimHei']
ax1 = fig1.add_subplot(111, projection='3d')

kmeans_members = kmeans_cluster_label == 0
s0 = x_h_w_50m_f[kmeans_members]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
kmeans_members = kmeans_cluster_label == 1
s1 = x_h_w_50m_f[kmeans_members]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y
kmeans_members = kmeans_cluster_label == 2
s2 = x_h_w_50m_f[kmeans_members]  # 给X,Y,Z赋值ndarray数组, 分别为C0*, w, y

cm1 = plt.cm.get_cmap('jet')  # 颜色映射，为jet型映射规则

fig1 = ax1.scatter3D(s0[:, 0], s0[:, 1], s0[:, 2], c=s0[:, 3], cmap=cm1, marker='*')
fig1 = ax1.scatter3D(s1[:, 0], s1[:, 1], s1[:, 2], c=s1[:, 3], cmap=cm1, marker='.')
fig1 = ax1.scatter3D(s2[:, 0], s2[:, 1], s2[:, 2], c=s2[:, 3], cmap=cm1, marker='v')
cb1 = plt.colorbar(fig1)  # 设置坐标轴
ax1.set_xlabel('身高')
ax1.set_ylabel('体重')
ax1.set_zlabel('50m')
cb1.ax.tick_params(labelsize=12)
cb1.set_label('肺活量', size=15)

plt.show()


#确定最优聚类数，将将结果可视化

plt.figure(6)  # 创建一个图
inertia_Je = []
k = []
#簇的数量
for n_clusters in range(1,10):
    cls = KMeans(n_clusters).fit(x_h_w_50m_f)
    inertia_Je.append(cls.inertia_)
    k.append(n_clusters)
plt.scatter(k, inertia_Je)
plt.plot(k, inertia_Je)
plt.xlabel("k")
plt.ylabel("Je")
plt.show()

