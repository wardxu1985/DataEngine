import pandas as pd

df = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/car_complain.csv')
print(df)

p = df['problem'].str.split(',', expand=True)
p = p.stack()
p = p.reset_index(level=1, drop=True)
p.name = 'problem'
df_new = df.drop(['problem'], axis=1).join(p)
print(df_new)
numble_1 = df_new.groupby(['brand'])['id'].nunique()
print('品牌投诉总数:\n', numble_1)
numble_2 = df_new.groupby(['car_model'])['id'].nunique()
print('车型投诉总数：\n', numble_2)
numble_3 = df.groupby(['brand'])['car_model'].nunique()
average = numble_1/numble_3
print('平均车型投诉数量：\n', average.sort_values(ascending=False))