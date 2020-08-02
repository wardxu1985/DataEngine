import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

pd.set_option('display.max_rows', None)


df = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/项目/project3/CarPrice_Assignment.csv')

# print(df.head(3))

train_x1 = df[['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem']]        #数据组1：文字类分组数据

train_x2 = df.drop(['fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem', 'car_ID', 'CarName'], axis=1)      #数据组2：数字类数据

le = preprocessing.LabelEncoder()           #使用LabelEncoder对分组数据进行数字化处理
train_x1 = train_x1.apply(le.fit_transform)     #对数据组1内的数据应用labelencoder

train_x = pd.concat((train_x1, train_x2), axis=1)       #数据组1与数据组2合并
# print(train_x.head(3))

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)        #数据进行归一化处理：数据范围0-1
train_x = min_max_scaler.fit_transform(train_x)     #对数据进行归一化


print(pd.DataFrame(train_x).head(3))

"""PCA主成分分析技术，对数据降维处理"""
def Pca(train_x):
    pca = sklearnPCA(n_components=2)        #pca降维参数：2维
    transformed = pd.DataFrame(pca.fit_transform(train_x))  #对数据组进行降维并转换成DateFrame
    # print(transformed)
    fig = plt.figure()                          #制图
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(transformed[0], transformed[1])
    plt.show()

"""使用sse簇内误方差 手肘法 确定分组系数K"""
def Sse(train_x):
    sse = []
    for k in range(1, 60):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(train_x)
        sse.append(kmeans.inertia_)
    x = range(1, 60)
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.plot(x, sse, 'o-')
    plt.show()


"""使用Si轮廓系数"""
def Si(train_x):
    Scores = []
    for k in range(2, 50):             # silhouette_score需要多个群集标签 k>1
        kmeans = KMeans(n_clusters=k)  # 构造聚类器
        kmeans.fit(train_x)
        Scores.append(silhouette_score(train_x, kmeans.labels_, metric='euclidean'))
    X = range(2, 50)
    plt.xlabel('K')
    plt.ylabel('SI')
    plt.plot(X, Scores, 'o-')
    plt.show()



Pca(train_x)
Sse(train_x)
Si(train_x)


kmeans = KMeans(n_clusters=15)              #通过手肘与轮廓系统确定k=15
kmeans.fit(train_x)                         #训练kmeans
predict_y = kmeans.predict(train_x)         #进行分组预测
result = pd.concat((df['CarName'], pd.DataFrame(predict_y)), axis=1)            #分组表格与原表格合并
result.rename({0: u'聚类结果'}, axis=1, inplace=True)                             #列重命名
# print(result.sort_values(by='聚类结果', ascending=False))
result = result.drop('CarName', axis=1).join(result['CarName'].str.split(' ', expand=True)[0].rename('company')).join(result['CarName'].str.split(' ', expand=True)[1].rename('car'))               #拆分carname组并重命名
print(result.sort_values(by='聚类结果', ascending=False))                       #按照聚类结果排序


"""显示包含volkswagen的聚类组"""
for i in range(1, 15):
    list_1 = result[result['聚类结果'] == i]
    ll = list_1['company'].values.tolist()
    if 'volkswagen' in ll:
        print("与volkswagen为竞品的车辆：")
        print(list_1)

"""使用pca降维绘制2D图，并按颜色分组"""
def Pca_fig2D():
    pca = sklearnPCA(n_components=2)        #pca降维参数：2维
    transformed = pd.DataFrame(pca.fit_transform(train_x))  #对数据组进行降维并转换成DateFrame
    result1 = pd.concat((transformed, pd.DataFrame(predict_y)), axis=1)
    result1.columns = ['0', '1', '2']
    # print(result1)
    # print(result1['0'], result1['1'])
    fig = plt.figure()                          #制图
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(result1['0'], result1['1'], c=result1['2'])      #根据分组显示颜色
    plt.show()

"""使用pca降维绘制3D图，并按颜色分组"""
def Pca_fig3D():
    pca = sklearnPCA(n_components=3)  # pca降维参数：3维
    transformed = pd.DataFrame(pca.fit_transform(train_x))  # 对数据组进行降维并转换成DateFrame
    result1 = pd.concat((transformed, pd.DataFrame(predict_y)), axis=1)
    result1.columns = ['0', '1', '2', '3']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=result1['0'], ys=result1['1'], zs=result1['2'], c=result1['3'])
    plt.show()


Pca_fig2D()
Pca_fig3D()