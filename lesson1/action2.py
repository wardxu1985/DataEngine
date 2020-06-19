import pandas as pd

data = {'姓名': ['张飞', '关羽', '刘备', '典韦', '许褚'], '语文': [68, 95, 98, 90, 80], '数学': [65, 76, 86, 88, 90], '英语': [30, 98, 88, 77, 90]}
df = pd.DataFrame(data)
df = df.set_index('姓名')
print(df)
print(df.describe())
df['总成绩'] = df.sum(axis=1)
print(df.sort_values(by='总成绩'))
