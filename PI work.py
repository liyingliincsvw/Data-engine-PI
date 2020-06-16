#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 作业1
sum = 0
number = 2
while number < 101:
    sum = sum + number
    number = number + 2
print(sum)


# In[72]:


# 作业2
import pandas as pd
from pandas import DataFrame
data = {'语文':[68, 95, 98, 90, 80], 
        '数学':[65, 76, 86, 88, 90], 
        '英语':[ 30, 98, 88 ,77 ,90]}
score = DataFrame(data, index = ['张飞','关羽', '刘备', '典韦', '许褚'])
print(score)
sumscore = score.sum(1)
print("三门课程的平均成绩分别为:" + "\n" ,score.mean(),sep = "")
print("三门课程的最小成绩分别为:" + "\n" ,score.min(),sep = "")
print("三门课程的最小成绩分别为:" + "\n" ,score.max(),sep = "")
print("三门课程的方差分别为:" + "\n" ,score.var(),sep = "")
print("三门课程的标准差分别为:" + "\n" ,score.std(),sep = "")
print("五个人的总成绩从高到低排名:" + "\n", sumscore.sort_values(ascending = False),sep = "")


# In[98]:


# 作业3
import pandas as pd
import numpy as np
data = pd.read_csv('D:\car_complain.csv')

# 品牌投诉总数
result1 = data.groupby(['brand'])['id'].agg(['count']).sort_values('count',ascending = False)
print(result1)

# 车型投诉总数
result2 = data.groupby(['car_model','brand'])['id'].agg(['count']).sort_values('count',ascending = False)
print(result2)

# 哪个品牌的平均车型投诉最多
result3 = data.groupby(['brand','car_model'])['id'].agg(['count']).groupby(['brand']).mean().sort_values('count',ascending = False)
print(result3)


# In[ ]:





# In[ ]:




