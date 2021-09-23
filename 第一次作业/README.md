## 模式识别与机器学习第一次作业-孙佳伟（202122060713）

[项目地址：https://github.com/sinary-sys/pattern_recognition/tree/master/%E7%AC%AC%E4%B8%80%E6%AC%A1%E4%BD%9C%E4%B8%9A](https://github.com/sinary-sys/pattern_recognition/tree/master/%E7%AC%AC%E4%B8%80%E6%AC%A1%E4%BD%9C%E4%B8%9A)

### 一、 以肺活量为例，画出男女生肺活量的直方图并做对比

#### 1、 表格数据的导入

使用`python`中的`pandas`库，`pandas`是专门为处理表格和混杂数据设计的。

```python
import pandas as pd
```

使用`pandas`库中的`read_excel`方法，将老师提供的excel表格读入。

```python
path = "F:\Mirror\学习资料\研一\pattern_recognition\第一次作业\作业数据_2021合成.xls"
data = pd.read_excel(path)
```

读入后的结果：

```python
      编号  性别 男1女0    籍贯  身高(cm)  体重(kg)  ...  喜欢颜色  喜欢运动  喜欢文学 喜欢数学  喜欢模式识别
0      1        1    湖北   163.0    51.0  ...     蓝     1     1  NaN     NaN
1      2        1    河南   171.0    64.0  ...     蓝     0     0  NaN     NaN
2      3        1    云南   182.0    68.0  ...     蓝     1     0  NaN     NaN
3      4        1    广西   172.0    66.0  ...     绿     0     1  NaN     NaN
4      5        1    四川   185.0    80.0  ...     蓝     0     0  NaN     NaN
..   ...      ...   ...     ...     ...  ...   ...   ...   ...  ...     ...
346  347        1  四川巴中   163.0    75.0  ...     蓝     0     0  NaN     NaN
347  348        1    北京   183.0    72.0  ...     白     0     0  NaN     NaN
348  349        1   内蒙古   170.0    60.0  ...     黄     1     0  NaN     NaN
349  350        1  四川巴中   168.0    55.0  ...     橙     1     0  NaN     NaN
350  351        1  湖南邵阳   168.0    50.0  ...     白     1     0  NaN     NaN

[351 rows x 13 columns]
```

```python
print(type(data))
```

```python
<class 'pandas.core.frame.DataFrame'>
```

读入的`data`是一个`DataFrame`类型的数据

- `DataFrame`是一个表格型的数据类型，每列值类型可以不同，是最常用的pandas对象。
- `DataFrame`既有行索引，也有列索引，它可以被看做由Series组成的字典（共用同一个索引）。
- `DataFrame`中的数据是以一个或多个二维块存放的（而不是列表、字典或别的一维数据结构）。

#### 2、导入数据的解析和肺活量绘图

```python
Vital_capacity = data['肺活量']

print(Vital_capacity)
```

将`肺活量`的列读取为`Vital_capacity`，将`性别`读入为`gender`

```python
gender = data['性别 男1女0']
```

使用`matplotlib`绘图库，`matplotlib`是一个用于创建出版质量图表的桌面绘图包（主要是2D方面）

导入`matplotlib`绘图库

```python
import matplotlib.pyplot as plt
```

