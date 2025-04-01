import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f, ttest_rel

path = "../practice-data/estimate-data/"
filename = "problem5_27.csv"
data = pd.read_csv(path + filename)

'''
1번 문제
강의 상류와 하류의 생물 다양성 점수에 차이가 있는지 검정하시오. 
(단, 같은 강에서 상류와 하류는 서로 독립적이지 않다.)

1. 귀무가설과 대립가설을 설정하시오.
2. 가설에 대한 검정 통계량과 유의 확률을 구하고, 
    귀무가설 기각 여부를 판단하시오
'''
# H0: 상류와 하류의 생물 다양성 점수가 같다.
# H1: 상류와 하류의 생물 다양성 점수가 다르다.
up = data['up']
down = data['down']

tstat, pval = ttest_rel(up, down, alternative='two-sided')
pval < 0.05  # False
# 대응표본 t-검정을 사용하여 상류와 하류의 생물 다양성 점수 차이를 검정.
# 귀무가설 기각 X
# 즉, 상류와 하류의 생물 다양성 점수는 차이가 없다.






filename = "problem5_32.csv"
data = pd.read_csv(path + filename)
'''
2번 문제
성별에 따른 급여의 평균 차이가 있는지 검정하시오.

1. 귀무가설과 대립가설을 설정하시오.
2. 성별과 급여의 차이를 알아보기 위해 데이터를 시각화하시오.
3. 그룹 간 분산의 차이가 있는지 검정하시오.
4. 가설에 대한 검정 통계량과 유의 확률을 구하고, 
    귀무가설 기각 여부를 판단하시오.
'''
# H0: 남성과 여성의 급여 평균은 같다.
# H1: 남성의 급여 평균이 더 높다.
male = data.loc[data["Gender"] == "Male", 'Salary']
female = data.loc[data["Gender"] == "Female", 'Salary']

# 박스플롯 시각화
plt.figure(figsize=(8, 6))
data.boxplot(column='Salary', by='Gender', grid=False, patch_artist=True)
plt.title('Salary Distribution by Gender')
plt.suptitle('')  # Remove the default title
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()



fstat = male.var(ddof=1) / female.var(ddof=1)
p_val = f.sf(fstat, dfn=len(male) - 1, dfd=len(female) - 1)
p_val < 0.05  # True
# 분산이 다르다.

tstat, pval = ttest_ind(male, female, alternative='greater', equal_var=False)
pval < 0.05  # True
# 귀무가설을 기각한다.
# 즉, 남성이 여성의 급여 평균보다 더 높다.



filename = "heart_disease.csv"
data = pd.read_csv(path + filename)
'''
3번 문제

심장 질환이 있는 그룹과 없는 그룹의 콜레스테롤 수치의 평균에 차이가 있는지 검정하시오.

1. 귀무가설과 대립가설을 설정하시오.
2. 변수 간 관계를 알아보기 위해 데이터를 시각화하시오.
3. 그룹 간 분산의 차이가 있는지 검정하시오.
4. 가설에 대한 검정 통계량과 유의 확률을 구하고, 대립가설 채택 여부를 판단하시오.

'''
# H0: 심장 질환이 있는 그룹과 없는 그룹의 콜레스테롤 수치의 평균은 같다.
# H1: 심장 질환이 있는 그룹이 콜레스테롤 수치 평균이 더 높다.
data.loc[]