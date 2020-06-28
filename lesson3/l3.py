from sklearn.cluster import KMeans, AgglomerativeClustering
import sklearn
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/lesson3/car_data.csv', encoding='GBK')
train_x = data[['人均GDP', '城镇人口比重', '交通工具消费价格指数', '百户拥有汽车量']]
# print(train_x)
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
# print(train_x)
kmeans = KMeans(n_clusters=5)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# print(predict_y)
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
# print(result)
result.rename(columns={0: '聚合'}, inplace=True)
print(result)


sse = []
for k in range(1, 11):
    # kmeans算法
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    # 计算inertia簇内误差平方和
    sse.append(kmeans.inertia_)

x = range(1, 11)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()


for i in range(5):
    s = result[result.聚合 == i]['地区']
    for area in s:
        print(area)
    print('\n')


from scipy.cluster.hierarchy import dendrogram, ward

model = AgglomerativeClustering(linkage='ward', n_clusters=5)
y = model.fit_predict(train_x)
print(y)
linkage_matrix = ward(train_x)
dendrogram(linkage_matrix)
plt.show()