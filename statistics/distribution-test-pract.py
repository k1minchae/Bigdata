import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f, ttest_rel, anderson, shapiro
import scipy.stats as stats
import seaborn as sns


'''
[연습문제] 분포 비교 방법 이해하기

'''
path = "../practice-data/estimate-data/"
filename = "problem5_32.csv"
dat = pd.read_csv(path + filename)

# 문제 1
# 성별에 따른 급여의 분포가 정규분포를 따르는지 확인하고자 한다.

# 1. Q-Q plot을 그리시오.
# 2. 정규성 검정을 통해 시각화 결과와 일치하는지 확인하시오.
# 3. 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.

dat.isna().sum()  # 결측치 확인


male = dat.loc[dat['Gender'] == 'Male', 'Salary']
female = dat.loc[dat['Gender'] == 'Female', 'Salary']

# 1. Q-Q plot 그리기
stats.probplot(male, dist="norm", plot=plt)
plt.title('Male QQ Plot')
plt.grid()
plt.show()

stats.probplot(female, dist="norm", plot=plt)
plt.title('Female QQ Plot')
plt.grid()
plt.show()

# 2. 정규성 검정
mt, m_pval = shapiro(male)
ft, f_pval = shapiro(female)

m_pval < 0.05  # False
# 남성의 p-value가 0.05보다 크므로 귀무가설을 기각하지 않는다.
# 즉, 남성은 정규성을 만족한다.

f_pval < 0.05  # False
# 여성의 p-value가 0.05보다 크므로 귀무가설을 기각하지 않는다.
# 즉, 여성은 정규성을 만족한다.




'''
문제 2

1. Q-Q plot을 그리시오.
2. 정규성 검정을 통해 시각화 결과와 일치하는지 확인하시오.
3. 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.

'''

filename = "heart_disease.csv"
dat = pd.read_csv(path + filename)
dat.isna().sum()  # 결측치 확인

# 결측치 제거
non_disease = dat.loc[dat['target'] == 'no', 'chol'].dropna()
disease = dat.loc[dat['target'] == 'yes', 'chol'].dropna()


# 1. Q-Q plot 그리기
stats.probplot(disease, dist="norm", plot=plt)
plt.title('Disease QQ Plot')
plt.grid()
plt.show()

stats.probplot(non_disease, dist="norm", plot=plt)
plt.title('Non-disease QQ Plot')
plt.grid()
plt.show()


# 정규성 검정
disease.size, non_disease.size  # 165, 137

# size 가 50 이상이므로 Shapiro-Wilk test를 사용하지 않는다.
# Anderson-Darling test를 사용한다.
disease_result = anderson(disease, dist='norm')
non_disease_result = anderson(non_disease, dist='norm')
disease_result.statistic < disease_result.critical_values[2]  # False
non_disease_result.statistic < non_disease_result.critical_values[2]  # True
# 질병이 있는 그룹은 정규성을 만족하지 않는다.
# 질병이 없는 그룹은 정규성을 만족한다.


# Shapiro-Wilk test
# H0: 데이터가 정규분포를 따른다.
# H1: 데이터가 정규분포를 따르지 않는다.
# p-value가 0.05보다 작으면 귀무가설을 기각하고 데이터가 정규분포를 따르지 않는다고 판단한다.
# p-value가 0.05보다 크면 귀무가설을 채택하고 데이터가 정규분포를 따른다고 판단한다.
dt, d_pval = shapiro(disease)
nt, nd_pval = shapiro(non_disease)
d_pval < 0.05  # True
# 질병이 있는 그룹은 정규성을 만족하지 않는다.

nd_pval < 0.05  # False
# 질병이 없는 그룹은 정규성을 만족한다.



'''
문제 3

당뇨병 여부에 따른 두 그룹의 BMI의 분포가 정규분포를 따르는지 알아보려고 한다.

1. Q-Q plot을 그리시오.
2. 정규성 검정을 통해 시각화 결과와 일치하는지 확인하시오.
3. 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.

'''
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
dat.head()

dat.isna().sum()  # 결측치 확인

# 데이터 전처리
disease = dat.loc[dat['Outcome'] == 1, 'BMI']
non_disease = dat.loc[dat['Outcome'] == 0, 'BMI']

# Q-Q plot 그리기
stats.probplot(disease, dist="norm", plot=plt)
plt.title('Disease QQ Plot')
plt.grid()
plt.show()

stats.probplot(non_disease, dist="norm", plot=plt)
plt.title('Non-Disease QQ Plot')
plt.grid()
plt.show()


# 표본 크기 확인
disease.size, non_disease.size  # 268, 500
# 표본 크기가 50 이상이므로 Shapiro-Wilk test를 사용하지 않는다.
# Anderson-Darling test를 사용한다.

disease_result = anderson(disease, dist='norm')
non_disease_result = anderson(non_disease, dist='norm')

disease_result.statistic < disease_result.critical_values[2]  # False
non_disease_result.statistic < non_disease_result.critical_values[2]  # False
# 두 그룹 모두 정규성을 만족하지 않는다.

# Shapiro-Wilk test
# H0: 데이터가 정규분포를 따른다.
# H1: 데이터가 정규분포를 따르지 않는다.

dt, d_pval = shapiro(disease)
nt, nd_pval = shapiro(non_disease)
d_pval < 0.05  # True
nd_pval < 0.05  # True
# 두 그룹 모두 정규성을 만족하지 않는다.


'''
4번 문제

'''

