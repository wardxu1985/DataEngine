import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

pd.set_option('display.max_rows', None)


df = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/项目/project3/CarPrice_Assignment.csv')

# print(df.head(3))

train_x1 = df[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']]

train_x2 = df.drop(['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'car_ID', 'CarName'], axis=1)

le = preprocessing.LabelEncoder()
train_x1 = train_x1.apply(le.fit_transform)

train_x = pd.concat((train_x1, train_x2), axis=1)
# print(train_x.head(3))

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
train_x = min_max_scaler.fit_transform(train_x)


# print(pd.DataFrame(train_x).head(3))
#
# pca = sklearnPCA(n_components=2)
# transformed = pd.DataFrame(pca.fit_transform(train_x))
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(transformed[0], transformed[1])
#
# plt.show()
#
#
# sse = []
# for k in range(1, 30):
#     kmeans = KMeans(n_clusters=k)
#     kmeans.fit(train_x)
#     sse.append(kmeans.inertia_)
# x = range(1, 30)
# plt.xlabel('K')
# plt.ylabel('SSE')
# plt.plot(x, sse, 'o-')
# plt.show()


kmeans = KMeans(n_clusters=15)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
result = pd.concat((df['CarName'], pd.DataFrame(predict_y)), axis=1)
result.rename({0: u'聚类结果'}, axis=1, inplace=True)
# print(result.sort_values(by='聚类结果', ascending=False))
result = result.drop('CarName', axis=1).join(result['CarName'].str.split(' ', expand=True)[0].rename('company')).join(result['CarName'].str.split(' ', expand=True)[1].rename('car'))
print(result.sort_values(by='聚类结果', ascending=False))

