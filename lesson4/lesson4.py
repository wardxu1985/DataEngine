import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





dataset = pd.read_csv(r'/Users/wardxu/Documents/GitHub/DataEngine/lesson4/Market_Basket_Optimisation.csv', header = None)
pd.set_option('display.max_columns', None)

def rule1():
    from efficient_apriori import apriori
    transactions = []
    for i in range(len(dataset)):
        temp = []
        for j in range(dataset.shape[1]):
            if str(dataset.values[i, j]) != 'nan':
                temp.append(str(dataset.values[i, j]))
        transactions.append(temp)
    # print(transactions)

    itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
    print("频繁项集：", itemsets)
    print("关联规则：", rules)




def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def rule2():
    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules

    data = dataset.stack()
    #转置
    # print(data)

    a = data.reset_index()
    #更新index
    print(a)

    b = a.groupby(['level_0', 0])[0].count().unstack().fillna(0)
    #使用原订单序号与商品名称进行表格展开
    # print(b)

    itemsets = b.applymap(encode_units)
    #表格中数据转换成0/1

    print(itemsets)

    frequent_itemsets = apriori(itemsets, min_support=0.02, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

    print(frequent_itemsets.sort_values(by="support", ascending=False))
    print(rules.sort_values(by="lift", ascending=False))

rule1()
rule2()



