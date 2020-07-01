#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#数据加载
data = pd.read_csv('car_data.csv',encoding="GBK")
train_x = data[["地区", "人均GDP", "城镇人口比重", "交通工具消费价格指数", "百户拥有汽车量"]]
#print(train_x)

#将地区名称标准化
le = LabelEncoder()
train_x['地区'] = le.fit_transform(train_x['地区'])
#print(train_x)

#规范化到[0,1]空间
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
train_x = pd.DataFrame(train_x)
#print(train_x)


#使用手肘法确定K值
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    sse.append(kmeans.inertia_)
x = range(1, 11)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(x,sse,'o-')    
plt.show()


#使用KMeans聚类，K值定为4
kmeans = KMeans(n_clusters=4)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)

#将聚类结果插入原数据中
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0:u'聚类结果'}, axis=1, inplace=True)
print(result)
result.to_csv("car_data_result.csv", index=False)


#使用层次聚类
from scipy.cluster.hierarchy import dendrogram, ward
#from sklearn.cluster import KMeans, AgglomerativeClustering
#model = AgglomerativeClustering(linkage='ward', n_clusters=4)
#y = model.fit_predict(train_x)
#print(y)

linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




