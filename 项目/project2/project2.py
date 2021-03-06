from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

pd.set_option('display.max_columns', None)

df = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/项目/project2/订单表.csv', encoding='gbk')      #读取csv文件
# print(df)

df = df.groupby(['客户ID', '订单日期', '产品名称'])['产品名称'].count().unstack().fillna(0)       #使用一个用户一次购买的产品进行展开
# print(df)


def encode_units(x):        #把数字转化成0/1
    if x <= 0:
        return 0
    if x >= 1:
        return 1


df = df.applymap(encode_units)      #应用自定义函数
print(df)

itemsets = apriori(df, use_colnames=True, min_support=0.03)         #频繁项集>0.03
rules = association_rules(itemsets, metric='confidence', min_threshold=0.03)        #关联规则:置信度>0.03

print(itemsets.sort_values(by='support', ascending=False))
print(rules.sort_values(by='confidence', ascending=False))


