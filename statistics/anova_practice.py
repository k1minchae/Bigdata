'''
분산분석 이해하기 연습문제

'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False



# 문제 1
# 세 개의 지역에 따라 학습 프로그램 비용이 서로 다른지를 알아보고자 합니다.
data1 = {
    "Area": [1]*6 + [2]*6 + [3]*6,
    "Tuition": [6.2, 9.3, 6.8, 6.1, 6.7, 7.5, 7.5, 8.2, 8.5, 8.2, 7.0, 9.3, 5.8, 6.4, 5.6, 7.1, 3.0, 3.5]
}
df1 = pd.DataFrame(data1)
df1.head()

# 1. 데이터 시각화
sns.boxplot(x='Area', y='Tuition', data=df1, palette='Set2')
plt.title("지역별 교육비")
plt.xticks([0, 1, 2], ['지역1', '지역2', '지역3'])
plt.xlabel("")
plt.ylabel("교육비")
plt.show()

# 2. 유의수준 0.05에서 세 지역의 평균 학습 비용이 동일하다는 귀무가설을 검정하세요.
help(f_oneway)
area1 = df1[df1['Area'] == 1]['Tuition']
area2 = df1[df1['Area'] == 2]['Tuition']
area3 = df1[df1['Area'] == 3]['Tuition']

stat, pval = f_oneway(area1, area2, area3)
print(f"F 통계량: {stat:.4f}, p-value: {pval:.4f}")

pval < 0.05 # True

# 귀무가설 기각
# 세 지역의 평균 학습 비용은 동일하지 않다.


# 3. 사후검정을 통해 어떤 지역 간에 차이가 있는지 확인하세요.
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# 사후검정: Tukey's HSD
tukey = pairwise_tukeyhsd(endog=df1['Tuition'],     # 종속 변수
                          groups=df1['Area'],       # 그룹 변수
                          alpha=0.05)               # 유의수준

print(tukey) # 지역 2와 지역3의 차이가 유의미합니다.
# 지역 1과 3은 기각의 임계점에 걸려있습니다.
# 지역 1과 2는 유의미한 차이가 없습니다.



# 문제 2
# 세 가지 매장 유형(패션 매장, 음악 전문점, 스포츠 용품점)별로 
# 직원의 평균 연령에 차이가 있는지를 검정합니다.

anova_data2 = {
    "Store_Type": ["Fashion"]*6 + ["Music"]*6 + ["Sporting"]*6,
    "Age": [45, 47, 50, 48, 46, 49, 25, 27, 28, 26, 24, 29, 30, 31, 28, 32, 29, 27]
}
df2 = pd.DataFrame(anova_data2)
df2.head()

# 1. 데이터 시각화
sns.boxplot(x='Store_Type', y='Age', data=df2, palette='Set2')
plt.title("매장 유형별 직원 연령")
plt.xticks([0, 1, 2], ['패션 매장', '음악 전문점', '스포츠 용품점'])
plt.xlabel("")
plt.ylabel("직원 연령")
plt.show()


# 2. 유의수준 0.05에서 평균 연령의 차이를 검정하세요.
fashion = df2[df2['Store_Type'] == 'Fashion']['Age']
music = df2[df2['Store_Type'] == 'Music']['Age']
sport = df2[df2['Store_Type'] == 'Sporting']['Age']

stat, pval = f_oneway(fashion, music, sport)
print(f"F 통계량: {stat:.4f}, p-value: {pval:.4f}")

pval < 0.05 # True
# 귀무가설 기각
# 세 가지 매장 유형의 평균 연령은 동일하지 않다.

# 3. 유의미한 차이가 있다면, 사후검정을 실시하세요.
tukey = pairwise_tukeyhsd(endog=df2['Age'],     # 종속 변수
                          groups=df2['Store_Type'], # 그룹 변수
                          alpha=0.05)               # 유의수준
print(tukey) # 전부 차이가 있습니다.



# 문제 3
# 세 가지 식품 종류별로 1회 제공량당 지방 함량 평균이 다른지를 분석합니다.
data3 = {
    "Food_Type": ["Type1"]*6 + ["Type2"]*6 + ["Type3"]*6,
    "Fat_Content": [7.0, 6.5, 8.2, 6.8, 7.1, 6.9, 9.1, 8.7, 8.9, 9.3, 8.5, 8.8, 5.9, 6.1, 5.8, 6.0, 5.5, 6.2]
}
df3 = pd.DataFrame(data3)
df3.head()

# 1. 데이터를 시각화하세요.
sns.boxplot(x='Food_Type', y="Fat_Content", data=df3, palette='Set2')
plt.title("식품 종류별 지방 함량")
plt.xticks([0, 1, 2], ['식품 1', '식품 2', '식품 3'])
plt.xlabel("")
plt.ylabel("지방 함량")
plt.show()


# 2. 유의수준 0.05에서 식품 종류에 따른 지방 함량의 평균 차이를 검정하세요.
stat, pval = f_oneway(
    df3[df3['Food_Type'] == 'Type1']['Fat_Content'], # 식품 1
    df3[df3['Food_Type'] == 'Type2']['Fat_Content'], # 식품 2
    df3[df3['Food_Type'] == 'Type3']['Fat_Content']  # 식품 3
)

print(f"F 통계량: {stat:.4f}, p-value: {pval:.4f}")

pval < 0.05 # True
# 귀무가설 기각
# 식품 종류에 따라 지방 함량의 평균이 서로 다르다.


# 3. 차이가 있는 경우, 어떤 그룹 간에 차이가 있는지 분석하세요.
tukey = pairwise_tukeyhsd(endog=df3['Fat_Content'],     # 종속 변수
                          groups=df3['Food_Type'], # 그룹 변수
                          alpha=0.05)               # 유의수준
print(tukey)
# 전부 차이가 있습니다.