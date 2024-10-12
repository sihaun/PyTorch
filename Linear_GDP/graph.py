import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = 'False'

path_real_wages = 'real_wages.csv'
path_GDP = 'GDP.csv'
data1 = pd.read_csv(path_real_wages, encoding='UTF-8')
data2 = pd.read_csv(path_GDP, encoding='UTF-8')

year = data1.columns[1:]
#print(data1.iloc[1])

df = [data1.iloc[index] for index in range(2)]
df.append(data2.iloc[0])
#print(df)

df = pd.DataFrame(df).T[1:]
df.columns = ['실질임금_증가율', '노동생산성지수_증가율', '경제성장률']

df['연도'] = df.index
#print(df['연도'])


df.plot(x='실질임금_증가율', y='경제성장률', kind='scatter', color='red')
plt.xlabel('실질임금증가율')
plt.ylabel('경제성장율')
plt.show()