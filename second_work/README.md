## 模式识别与机器学习第二次作业

### 孙佳伟-202122060713



### 1.   采用C均值聚类算法对男女生样本数据中的身高、体重、50m成绩、肺活量4个特征进行聚类分析，考察不同的类别初始值以及类别数对聚类结果的影响，并以友好的方式图示化结果。

k-mean算法如下

LLoyd’s algorithm is iterative and decomposes the k-means problem into two distinct, alternating steps: the *assignment* step and the *update* step. The full algorithm can be described in pseudocode as follows:

1. Given cluster centroids μiμi initialized in some way,
2. For iteration $t=1..T$:
   1. Compute the distance from each point xx to each cluster centroid μμ,
   2. Assign each point to the centroid it is closest to,
   3. Recompute each centroid μ as the mean of all points assigned to it,

where T is the number of iterations we wish to run this algorithm for (typically a few hundred). In each iteration, (1) and (2) are the assignment step and (3) is the update step. The time complexity of this approach is $O(n×k×T)$. As you can see, this algorithm is very straightforward and simple. I’d like to emphasize that the first step, cluster initialization, is very important too and bad initialization can delay proper convergence by many iterations. 

下面是`k-mean`算法的伪代码

```c++
输入:样本集D={x_1,x_2,...,x_n};聚类簇数c
过程：
1:从D中随机选择c个样本作为初始均值向量{u_1，u_2,...,u_c}
2:repeat
3:令C_i=Ø(1≤i≤c)
4:for j = 1,...,n do
5:  计算样本x_j与各均值向量u_i(1≤i≤c)的距离：d_ji = ||x_j-u_i||2;
6:  根据距离最近的均值向量将x_j归入该簇
7:end for
8:for i = 1,...,c do
9:  计算新的均值向量u'_i
10: if u'_i ≠ u_i then
11:    将当前均值向量u_i更新为u'_i
12：   else
13:    保持当前均值向量不变
14：   end if
15:end for
16:until 当前所有均值向量不再更新
17：return 簇划分结果
输出：簇划分C={C_1,C_2,...,C_c}
```

本次聚类的样本`x`是一个四维的特征，先把四维特征提取出来，并且做一个展示，由于四维数据不便于展示，二维数据在空间中是一个点，三维数据在空间中有三维坐标，四维就很难可视化展示，先做一个关于二维的身高体重，直观验证一下k-mean算法







### 2.   采用分级聚类算法对男女生样本数据进行聚类分析。尝试采用身高，体重、50m、肺活量4个特征进行聚类，并以友好的方式图示化结果。

