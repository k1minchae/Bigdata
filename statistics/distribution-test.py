import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f, ttest_rel, norm
from scipy import stats as sp
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# p-th percentile 구하는 방법
# 1. 데이터 순서대로 정렬
# 2. (n - 1) * p + 1계산한다.
# 3. 정수부분은 j, 소수부분은 h로 한다.
# 4. p-th percentile 구하는 공식 적용
#    p-th percentile = x[j - 1] * (1 - h)  + h * x[j] (j는 1부터 시작)


# 예제
x = [155, 126, 27, 82, 115, 140, 73, 92, 110, 134]
x = np.array(x)
x.sort()

# 25% 백분위수
p = 0.25
n = len(x)
pos = (n - 1) * p + 1
j = int(np.floor(pos))
h = pos - j

percentile_25 = x[j - 1] * (1 - h) + h * x[j]  # 25% 백분위수
print(f"25% Percentile: {percentile_25}")  # 84.5 

# 50% 백분위수
p = 0.5
pos = (n - 1) * p + 1
j = int(np.floor(pos))
h = pos - j

percentile_50 = x[j - 1] * (1 - h) + h * x[j]  # 50% 백분위수
print(f"50% Percentile: {percentile_50}")   # 112.5

# 75% 백분위수
p = 0.75
pos = (n - 1) * p + 1
j = int(np.floor(pos))
h = pos - j

percentile_75 = x[j - 1] * (1 - h) + h * x[j]
print(f"75% Percentile: {percentile_75}")   # 132.0


# norm.ppf()를 이용해서 백분위수 구하면 쉬운데
# 어떤 분포를 따르는 지 모르기 때문에 percentile을 구할 수 없다.
# 그래서 percentile을 구하는 방법을 알아보았다.

# 하지만, 이 방법을 사용해서 파이썬에서 백분위수를 구한다는 것을 확인하고, 
# 실제 백분위수는 numpy의 percentile() 함수를 사용해서 구하면 됩니다.
np.percentile(x, 25)  # 84.5
np.percentile(x, 50)  # 84.5
np.percentile(x, 75)  # 84.5


# 나의 데이터가 정규분포를 따를까?
np.percentile(x, 25)  # 84.5
norm.ppf(0.25, loc=np.mean(x), scale=x.std(ddof=1))  # 79.8

np.percentile(x, 50)  # 112.5
norm.ppf(0.50, loc=np.mean(x), scale=x.std(ddof=1))  # 105.4

np.percentile(x, 75)  # 132.0
norm.ppf(0.75, loc=np.mean(x), scale=x.std(ddof=1))  # 131.0

plt.hist(x, density=True, bins=10, alpha=0.5, color='g')
k = np.linspace(30, 170, 10000)
plt.plot(k, norm.pdf(k, loc=np.mean(x), scale=x.std(ddof=1)), color='r')
plt.title('내 데이터와 정규분포 비교하기')
plt.xlabel('값')
plt.ylabel('밀도')
plt.legend(['정규분포', '내 데이터'])
plt.show()



# 데이터랑 이론값이랑 가깝게 찍히면 정규분포를 따른다고 볼 수 있다.
# qqplot을 그려보면 더 확실하게 알 수 있다.
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55,
                    3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87])

# qqplot 그리기
sp.probplot(data_x, dist="norm", plot=plt)
plt.title('QQ Plot')
plt.xlabel("이론 분포의 분위수")
plt.ylabel("실제 데이터의 분위수")
plt.grid()
plt.show()


# histogram 그리기 => 데이터 수가 적어서 정확하지 않다.
plt.hist(data_x, density=True, bins=10, alpha=0.5, color='g')
k = np.linspace(-10, 20, 10000)
plt.plot(k, norm.pdf(k, loc=np.mean(data_x), scale=data_x.std(ddof=1)), color='r')
plt.title('Sample Data and Norm Distribution')
plt.xlabel('V')
plt.ylabel('D')
plt.legend(['norm', 'sample'])
plt.show()



# Shapiro-Wilk test
# 정규성 검정 방법 중 하나로, 귀무가설은 데이터가 정규분포를 따른다는 것이다.
# p-value가 0.05보다 작으면 귀무가설을 기각하고 데이터가 정규분포를 따르지 않는다고 판단한다.
# p-value가 0.05보다 크면 귀무가설을 채택하고 데이터가 정규분포를 따른다고 판단한다.
# 정규성 검정은 데이터의 크기가 작을 때 유용하다. (n < 50)
# 단점: 민감해서 p-value가 작게 나올 수 있다.
# qqplot이랑 함께 사용하자.


# w = 0 ~ 1
# 1에 가까울 수록 정규분포를 따른다.
# w = (이론적인 값 * X[i]) / (표본 분산: 내가 가진 정보)
w, pval = sp.shapiro(data_x)
print(f"statistic: {w}, p-value: {pval}")
pval < 0.05   # False
# p-value: 0.05보다 크므로 귀무가설을 채택한다.
# 데이터가 정규분포를 따른다.



# 데이터를 사용해서 누적분포함수 그리기
# 데이터에 들어있는 분포 정보를 사용해서 누적 분포함수 구하는 함수
from statsmodels.distributions.empirical_distribution import ECDF
data_x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11,
                11.97, 2.16, 3.24, 10.91, 11.36, 0.87])
ecdf = ECDF(data_x)
x = np.linspace(min(data_x), max(data_x))
y = ecdf(x)
plt.plot(x, y, marker='_', linestyle='none')

k = np.linspace(-2, 13, 100)
cdf_k = norm.cdf(k, loc=data_x.mean(), scale=data_x.std(ddof=1))
plt.plot(k, cdf_k, color='r')

plt.title("추정된 누적분포함수")
plt.xlabel("X축")
plt.ylabel("ECDF")
plt.show()





# 앤더슨-달링(Anderson-Darling) 검정
# 이론적인 누적분포함수와 데이터에서 뽑혀진 누적분포함수가 얼마나 비슷한지 체크하여 검정하는
# 방법입니다. 이 검정의 귀무가설과 대립가설은 다음과 같습니다.
# H0: 데이터가 특정 분포를 따른다. (보통은 정규분포에서 많이씀)
# H1: 데이터가 특정 분포를 따르지 않는다.
# 이 검정은 데이터의 크기에 상관없이 사용할 수 있습니다.

from scipy.stats import anderson
result = anderson(data_x, dist='norm')
print(f"검정통계량: {result.statistic:.2f}, 유의수준: {result.significance_level}, 임계값: {result.critical_values}")
# 유의수준 5일때 p-value가 0.679이므로 귀무가설 기각X
# 데이터가 정규분포를 따른다.

