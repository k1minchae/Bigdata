import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, t, binom, chi2, f

# U(3, 7)
uniform.mean(loc=3, scale=4) # 5.0
uniform.rvs(loc=3, scale=4, size=100).mean() # 5.33
uniform.rvs(loc=3, scale=4, size=10000).mean() # 5.007

# 샘플의 크기가 커질수록 모집단의 평균에 가까워짐


# 정규분포에서 95% 신뢰구간을 구하는 방법
# 왼쪽 기각역
norm.ppf(0.025, loc=13, scale=1.94) # 9.1976

# 오른쪽 기각역
norm.ppf(0.975, loc=13, scale=1.94) # 16.8023



# 95% 신뢰구간을 구하는 방법
uniform(loc=3, scale=4).var() # 1.33333
n = 20
x = uniform.rvs(size=n, loc=3, scale=4)

# 1. 표본평균 구하기
x_bar = np.mean(x)  # 4.822

# 2. 표본의 분산 구하기 (모분산을 알 때)
sigma = np.sqrt(uniform(loc=3, scale=4).var() / n)

# 3. 왼쪽 기각역 구하기
left = norm.ppf(0.025, loc=x_bar, scale=sigma)

# 4. 오른쪽 기각역 구하기
right = norm.ppf(0.975, loc=x_bar, scale=sigma)

# 5. 신뢰구간 구하기
(left, right)


# 과연 위 신뢰구간은 언제나 모평균을 포함하고있을까?
# 1000번 반복해서 신뢰구간을 구해보자
times = 1000
cnt = 0
for time in range(times):
    x = uniform.rvs(size=n, loc=3, scale=4)
    x_bar = np.mean(x)
    sigma = np.sqrt(uniform(loc=3, scale=4).var() / n)

    # 왼쪽 기각역
    left = norm.ppf(0.025, loc=x_bar, scale=sigma)

    # 오른쪽 기각역
    right = norm.ppf(0.975, loc=x_bar, scale=sigma)

    if left < 5 and right > 5:
        cnt += 1
print(cnt / times)  # 0.956
# 95.6%의 확률로 모평균을 포함하고 있다.


# 이거를 numpy로 바꿔보자
times = 1000
x = uniform.rvs(size=(times, n), loc=3, scale=4)  # (1000, 20) 샘플 생성
x_bar = np.mean(x, axis=1)  # 각 샘플의 표본평균 계산
sigma = np.sqrt(uniform(loc=3, scale=4).var() / n)  # 표본 표준편차

# 왼쪽 및 오른쪽 기각역 계산
left = norm.ppf(0.025, loc=x_bar, scale=sigma)
right = norm.ppf(0.975, loc=x_bar, scale=sigma)

# 모평균(5)이 신뢰구간에 포함되는지 확인
cnt = np.sum((left < 5) & (right > 5))
print(cnt / times)  # 0.951
# 95.1%의 확률로 모평균을 포함하고 있다.


# Z ~ N(0, 1^2)
# 를 X ~ N(3, 5^2)로 변환하는 방법
# 5Z + 3 = X
# 공식: X = μ + σZ

# 반대로 표준 정규분포로 변환하는 방법
# Z = (X - μ) / σ


# X의 표본을 표준정규분포로 변환하는 방법
# Z = (X^ - μ) / σ/sqrt(n)
# X^ = x_bar



'''
표본 평균이 4.952
모분산이 1.33333 일때 
모평균에 대한 86% 신뢰구간을 구하세요.

'''
x_bar = 4.952
sigma = np.sqrt(1.33333 / 20)
alpha = 0.86

# 표준 정규분포로부터 구하는 방법
z_007 = norm.ppf(0.93, loc=0, scale=1)

x_bar - sigma * z_007
x_bar + sigma * z_007

print(f'{left:.2f}, {right:.2f}') # 4.570952871335121, 5.333047128664879


'''
모분산을 모를 때 구하는법
시그마를 표본 표준편차로 대체한다. (n >= 30일 때)

단, n이 30보다 작을 때는 t 분포를 사용한다.

'''

# 귀무가설이란?
# 모집단의 모평균이 5이다.

# 대립가설이란?
# 모집단의 모평균이 5가 아니다.
# 대립가설을 검정하기 위해서는 t 분포를 사용한다.




# p-value 란?
# 귀무가설이 맞다는 가정하에 관측된 데이터보다 극단적인 데이터가 나올 확률

# 귀무가설: 평균이 100이다.
# 실제로 나온 평균: 105 (내가 관측한 값)
# 그럼 "평균이 100인데 실제로 내가 관측한 값이 나올 확률이 얼마나 되냐?" 이게 p-value야

# 신뢰구간이 95%라고 하면, p-value가 0.05보다 작으면 귀무가설을 기각할 수 있다.
# 이유: 내가 관측한 값이 나올 확률이 5%보다 작으니까, 평균이 100이라는 가정은 맞지 않다.
# 즉, p-value가 작으면 귀무가설을 기각할 수 있다.
# p-value가 작다는건, 관측된 데이터가 귀무가설과 맞지 않다는 뜻이니까.

# 보통 α = 0.05 (5%)로 둬

# → 이 말은: “귀무가설이 맞을 때, 5% 정도는 우리가 틀릴 수도 있어. 그 정도는 감수한다.”


'''
H0: 모평균이 4이다.

(모분산은 2이다.)

유의수준 알파는 3% 이다.

'''
x = np.array([6.663, 5.104, 3.026, 6.917, 5.645, 4.138, 4.058, 6.298, 6.506])

# 1. 검정통계량
sigma = np.sqrt(2 / len(x))  # 모분산을 알 때
z = (x.mean() - 4) / sigma  # 검정통계량

# 2. p-value?
p_value = 1 - norm.cdf(z, loc=0, scale=1)
p_value *= 2  # 양측검정이니까 2를 곱해준다.

# 판단?
p_value < 0.03  # True
# p-value가 0.03보다 작으니깐 귀무가설을 기각할 수 있다.
# 즉, 모평균이 4가 아니다.


'''
문제 연습

어느 커피숍에서 판매하는 커피 한잔의 평균 온도가 75도씨라고 주장하고 있습니다.
이 주장에 의문을 가진 고객이 10잔의 커피 온도를 측정한 결과 
다음과 같은 값을 얻었습니다.

[72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1]
단, 모표준편차는 1.2 입니다.
'''
x = np.array([72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1])

# 1. 귀무가설, 대립가설을 설정하세요.
# 귀무가설: 커피한잔의 평균 온도가 75도씨이다.
# 대립가설: 커피한잔의 평균 온도가 75도씨가 아니다.

# 2. 유의수준 5%에서 검정통계량 (z)을 구하세요.
z = (x.mean() - 75) / (1.2 / np.sqrt(10))

# 3. 유의 확률 값을 계산하세요.
p_value = norm.sf(abs(z), loc=0, scale=1)
p_value = norm.sf(abs(z))    # 표준 정규분포는 loc, scale을 생략할 수 있다.
p_value = norm.cdf(z)
p_value = min(1, p_value * 2)

# 4. 통계적 판단은?
p_value < 0.05  # False
# p-value가 0.05보다 크니깐 귀무가설을 기각할 수 없다.
# 즉, 커피한잔의 평균 온도가 75도씨라고 주장할 수 있다.

z_vals = np.linspace(-4, 4, 1000)
pdf_vals = norm.pdf(z_vals)

plt.figure(figsize=(10, 5))
plt.plot(z_vals, pdf_vals, label='Standard Normal Distribution')

# z 점 위치 선
plt.axvline(z, color='red', linestyle='--', label=f'z = {z:.2f}')
plt.axvline(-z, color='red', linestyle='--')  # 양측검정이니까 -z도 표시

# p-value 영역 색칠 (양측검정)
z_abs = abs(z)
x_shade_right = np.linspace(z_abs, 4, 100)
x_shade_left = np.linspace(-4, -z_abs, 100)
plt.fill_between(x_shade_right, norm.pdf(x_shade_right), color='red', alpha=0.3)
plt.fill_between(x_shade_left, norm.pdf(x_shade_left), color='red', alpha=0.3)


# 텍스트 추가
plt.text(z + 0.1, 0.02, f'z = {z:.2f}', color='red')
plt.title('Z-test Visualization with p-value Region (Two-tailed)')
plt.xlabel('Z-value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()


# 새로운 분포의 특성: t분포
# import t

# 자유도는 n - 1 이다.
# (t분포는, 모수가 1개: 자유도)
# (정규분포는, 모수가 2개: 모평균, 모분산)

k = np.linspace(-3, 3, 100)

# 자유도에 따른 t 분포의 모양 확인
plt.figure(figsize=(8, 6))
for df in [1, 3]:  # 다양한 자유도
    plt.plot(k, t.pdf(k, df=df), label=f'df={df}')

# 표준 정규분포와 비교
plt.plot(k, norm.pdf(k), label='Normal Distribution', linestyle='--', color='black')

plt.title('t-Distribution with Different Degrees of Freedom')
plt.xlabel('k')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()




# t 분포는 정규분포와 비슷하지만, 표본의 크기가 작을 때 더 넓은 분포를 가진다.
# t 분포는 자유도에 따라 모양이 달라진다.
# 자유도가 클수록 정규분포에 가까워진다.
# 자유도가 작을수록 t 분포는 더 넓고 뾰족하다.
# t 분포는 표본의 크기가 작을 때(<30), 모분산을 모르는 경우에 사용한다.



'''
문제1
한 공장에서 생산된 배터리의 평균 수명은 500시간이며 표준편차는50시간입니다 
이 배터리에서 100개의 표본을 추출했을 때 표본 평균의 분포에 대한 다음 질문에 답하시오
1. 표본 평균의 분포는 어떤 분포를 따르나요
2. 표준오차를 구하시오
3. 표본 평균이 510시간 이상일 확률을 구하시오

'''
# 1. 표본 평균의 분포는 어떤 분포를 따르나요
# 정규분포를 따른다.

# 2. 표준오차를 구하시오.
표준오차 = 50 / np.sqrt(100)

# 3. 표본 평균이 510시간 이상일 확률을 구하시오.
답 = norm.sf(510, loc=500, scale=표준오차)



'''
문제2
한 제품의 불량률이5%인 경우 이 제품20개를 무작위로 뽑았을 때
1. 불량품이 정확히2개 나올 확률을 구하시오
2. 불량품이2개 이하로 나올 확률을 구하시오
3. 불량품이3개 이상 나올 확률을 구하시오
'''

# 1. 불량품이 정확히 2개 나올 확률을 구하시오.
binom.pmf(k=2, n=20, p=0.05)

# 2. 불량품이 2개 이하로 나올 확률을 구하시오.
binom.cdf(k=2, n=20, p=0.05)

# 3. 불량품이 3개 이상 나올 확률을 구하시오.
binom.sf(k=2, n=20, p=0.05)



'''
문제3
한 학생의 수학 점수가 평균75점 표준편차8점인 정규분포를 따릅니다
1. 이 학생의 점수가85점 이상일 확률을 구하시오
2. 점수가70점과 80점 사이에 있을 확률을 구하시오
3. 상위10%에 해당하는 점수 기준 컷오프을 구하시오
'''

# 1. 이 학생의 점수가 85점 이상일 확률을 구하시오.
norm.sf(85, loc=75, scale=8)

# 2. 점수가 70점과 80점 사이에 있을 확률을 구하시오.
norm.cdf(80, loc=75, scale=8) - norm.cdf(70, loc=75, scale=8)

# 3. 상위 10% 에 해당하는 점수 기준
norm.ppf(0.9, loc=75, scale=8)


'''
문제5
한 제과점에서 하루 평균 판매되는 케이크의 개수가 50개라고 알려져 있습니다 
최근 데이터에서 표본40일 동안 평균 53개의 케이크가 판매되었고 
표준편차는 8개였습니다

1. 귀무가설과 대립가설을 설정하시오
2. 유의수준0.05 에서 z검정을 수행하시오
3. p-value를 계산하고 귀무가설을 기각할 수 있는지 판단하시오

'''
# 귀무가설: 하루 평균 판매되는 케이크의 개수가 50개이다.
# 대립가설: 하루 평균 판매되는 케이크의 개수가 50개가 아니다.
# 유의수준: 0.05

z = (53 - 50) / (8 / np.sqrt(40))  # 검정통계량
p_value = norm.sf(abs(z)) * 2
p_value < 0.05 # True
# p-value가 0.05보다 작으니깐 귀무가설을 기각할 수 있다.


##############################################################################
# 이틀차

# X의 기대값 E[X^4 + X^2 - 3] 추정하기
# sum(x^4 + x**2 - 3) / len(n)


# U(3, 7)
uniform(loc=3, scale=4).var() # 1.33333

n = 100
x = uniform.rvs(size=n, loc=3, scale=4)
x_bar = np.mean(x)

# 추정한 분산
var_bar = np.sum((x - x_bar)**2) / n


'''
스튜던트 정리

주어진 검정통계량이 t분포를 따르는지?
t = (X_bar − μ) / (S / sqrt(n))

'''
# X_bar 는 평균이 μ, 분산이 σ²/n인 정규분포를 따른다.
# ★ (n-1) * S² / σ² 는 카이제곱 분포를 따른다 (자유도 n-1)
# X_bar와 S²는 독립이다.
# ★ 그래서 (X_bar - μ) / (S / sqrt(n)) 은 t(n-1) 분포를 따른다.

# μ: 10, σ: 2, n: 17
# student t 정리 : S²는 카이제곱 분포를 따른다 (자유도 n-1) 확인
mu = 10
sigma = 2
n = 17
data_size = 1000
x = norm.rvs(loc=mu, scale=sigma, size=(data_size, n))
result = (n - 1) * x.var(ddof=1, axis=1) / sigma**2

# 카이제곱 분포를 따르는지 시각화
# 히스토그램: 샘플 데이터
plt.hist(result, bins=30, density=True, alpha=0.6, color='blue', label='Sample Data')

# 카이제곱 분포 PDF
x_vals = np.linspace(chi2.ppf(0.001, df=n-1), chi2.ppf(0.999, df=n-1), data_size)
plt.plot(x_vals, chi2.pdf(x_vals, df=n-1), 'r-', label=f'Chi-squared PDF (df={n-1})')

plt.title('Chi-squared Distribution vs Sample Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()

# t분포를 따르는지 확인
# ★ (X_bar - μ) / (S / sqrt(n)) 은 t(n-1) 분포를 따른다.
# 표본평균과 표본표준편차 계산
x_bar = x.mean(axis=1)
s = x.std(axis=1, ddof=1)

# t 통계량 계산
t_sample = (x_bar - mu) / (s / np.sqrt(n))

# t분포를 따르는지 시각화
# 히스토그램: 샘플 데이터
plt.hist(t_sample, bins=30, density=True, alpha=0.6, color='blue', label='Sample Data')

# t 분포 PDF
x_vals = np.linspace(t.ppf(0.001, df=n-1), t.ppf(0.999, df=n-1), data_size)
plt.plot(x_vals, t.pdf(x_vals, df=n-1), 'r-', label=f't PDF (df={n-1})')

plt.title('t-Distribution vs Sample Data')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()


# t검정통계량 계산 예제
x = np.array([4.62, 4.09, 6.2, 8.24, 0.77, 5.55, 3.11, 11.97, 2.16, 3.24, 10.91, 11.36, 0.87, 9.93, 2.9])
n = len(x)
# H0: 모평균이 7이다.

# t 검정통계량 계산
t_stat = (x.mean() - 7) / (x.std(ddof=1) / np.sqrt(n))

# p-value 계산
p_value = t.cdf(t_stat, df=n-1)
p_value = min(1, p_value * 2)  # 양측검정이니까 2를 곱해준다.

p_value < 0.05  # False
# p-value가 0.05보다 작지 않으므로 귀무가설을 기각할 수 없다.
# 즉, 모평균이 7이라고 주장할 수 있다.


# t 검정통계량 시각화
k = np.linspace(-4, 4, 1000)
plt.plot(k, t.pdf(k, df=n-1), label='t-distribution', color='blue')
plt.axvline(t_stat, color='red', linestyle='--', label=f't-statistic = {t_stat:.2f}')

# 기각 영역을 fill_between으로 시각화
critical_value = t.ppf(0.975, df=n-1)  # 양측검정, 유의수준 0.05
plt.fill_between(k, t.pdf(k, df=n-1), where=(k <= -critical_value) | (k >= critical_value), color='red', alpha=0.3, label='Rejection Region')

plt.title('t-test Statistic Visualization')
plt.xlabel('t-value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.show()






# 문제 4
x = np.array([72.4, 74.1, 73.7, 76.5, 75.3, 74.8, 75.9, 73.4, 74.6, 75.1])
n = len(x)
# H0: 모평균이 75도씨이다.
# H1: 모평균이 75도씨가 아니다.

t_stat = (x.mean() - 75) / (x.std(ddof=1) / np.sqrt(n))
p_value = t.cdf(t_stat, df=n-1)  # p-value 계산
p_value = min(1, p_value * 2)  # 양측검정이니까 2를 곱해준다.

p_value < 0.05  # False
# p-value가 0.05보다 작지 않으므로 귀무가설을 기각할 수 없다.
# 즉, 모평균이 75도씨라고 주장할 수 있다.

# t 검정통계량 시각화
k = np.linspace(-4, 4, 1000)
plt.plot(k, t.pdf(k, df=n-1), label='t-distribution', color='blue')
plt.axvline(t_stat, color='red', linestyle='--', label=f't-statistic = {t_stat:.2f}')
critical_value = t.ppf(0.975, df=n-1)  # 양측검정, 유의수준 0.05
plt.fill_between(k, t.pdf(k, df=n-1), where=(k <= -critical_value) | (k >= critical_value), color='red', alpha=0.3, label='Rejection Region')
plt.xlabel('t-value')
plt.ylabel('Density')
plt.legend()
plt.grid()
plt.title('t-test Statistic Visualization')
plt.show()


# t 검정의 기본적인 자료 형태는 다음과 같이 데이터가 
# 벡터 형태로 모든 표본이 같은 그룹으로 묶일 수 있는 형태입니다.

# 학생_ID 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
# 성적90, 85, 88, 92, 95, 80, 78, 85, 90, 95, 88, 92
# 성별 남, 여, 남, 여, 남, 여, 남, 여, 남, 여, 남, 여 (범주형 변수)

# 이런 형태로 데이터가 주어지면, t 검정을 수행하기 위해서는
# 성별에 따라 데이터를 분리해야 합니다.
# 즉, 남학생과 여학생의 성적을 각각 따로 분리해서 t 검정을 수행해야 합니다.
# 이런 경우에는 독립 2 표본 t 검정을 수행합니다.
# 귀무가설: 남학생과 여학생의 성적 차이가 없다.
# 대립가설: 남학생 그룹이 여학생 그룹의 성적보다 높다.


# 짝이 지어질 수 있는 데이터
# 시간 변수가 추가되어 교육 전후의 성적을 비교하는 경우
# 예를 들어, 학생 A의 교육 전 성적과 교육 후 성적을 비교하는 경우
# 이런 경우에는 t 검정을 수행하기 위해서는 성적을 짝지어야 합니다.
# 즉, 학생 A의 교육 전 성적과 교육 후 성적을 짝지어서 t 검정을 수행해야 합니다.
# 귀무가설: 교육 전과 후의 성적 차이가 없다.
# 대립가설: 교육 전과 후의 성적 차이가 있다.
# 이런 경우에는 대응 표본 t 검정을 수행합니다.
# 귀무가설: 교육 전과 후의 성적 차이가 없다.
# 대립가설: 교육 후에 성적이 높아졌다.

# 독립 2 표본 t 검정과 대응 표본 t 검정의 차이점은
# 독립 2 표본 t 검정은 두 그룹의 성적을 비교하는 것이고
# 대응 표본 t 검정은 한 그룹의 성적을 비교하는 것입니다.


'''
검정 선택 시 고려사항
t 검정의 형태를 결정할 때, 다음 두 가지를 고려 후 판단합니다.
1. 그룹 변수가 존재하는가?
2. 표본들을 짝지을 수 있는 특정 변수가 존재하는가?

위의 질문에 대한 답으로 그룹 변수가 존재하는 경우, 
데이터의 집단을 그룹 변수의 값에 따라서 2개로 나누어 생각할 수 있는 경우 
2 표본 t 검정을 선택합니다.

특정 데이터의 경우, 나뉘어진 표본들을 짝을 지을 수 있는 경우가 존재하는데, 
이 경우 대응 표본 t검정을 선택하고, 그렇지 않은 경우, 독립 2 표본 t 검정을 선택합니다.

'''

# T검정을 할 수 있는 상황/조건
# 1. 모집단이 정규분포를 따른다.
# 2. 모집단의 분산이 동일하다. (등분산성)
# 3. 표본의 크기가 작을 경우 검정의 신뢰도가 떨어질 수 있음.


from scipy.stats import ttest_1samp

# 독립 1표본 t 검정 - 모듈 사용
# H0: 모평균이 10이다.
x = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45])

t_stat, p_value = ttest_1samp(x, popmean=10, alternative="two-sided")  # x: 표본, 7: 모평균
(1 - t.cdf(t_stat, df=len(x)-1)) * 2  # p-value 계산
p_value < 0.05  # False
# p-value가 0.05보다 작지 않으므로 귀무가설을 기각할 수 없다.
# 즉, 모평균이 10이라고 주장할 수 있다.




# 독립 2표본 t 검정 - 모듈 사용
# H0: 남학생과 여학생의 성적 차이가 없다.
# H1: 남학생 그룹이 여학생 그룹의 성적보다 높다.
from scipy.stats import ttest_ind

sample = [9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52, 14.83, 13.03, 16.46, 10.84, 12.45]
gender = ["F"]*7 + ["M"]*5
my_tab2 = pd.DataFrame({"score": sample, "gender": gender})
male = my_tab2.loc[my_tab2["gender"] == "M", "score"]
female = my_tab2.loc[my_tab2["gender"] == "F", "score"]

t_stat2, p_value2 = ttest_ind(male, female, 
                              equal_var=False, # 동분산 가정 (분산이 같다는 가정)
                              alternative="greater")  # 첫번째 입력값 평균이 더 크다

# F 검정
# 귀무가설: 두 그룹의 분산이 같다.
# 대립가설: 두 그룹의 분산이 다르다.
from scipy.stats import f
f_stat = male.var(ddof=1) / female.var(ddof=1)  # F 검정 통계량
p_val = 1 - f.cdf(f_stat, dfn=len(male) - 1, dfd = len(female))
p_val *= 2
# p-value 계산

help(f)



# 대응 표본 t 검정 - 모듈 사용
# H0: 교육 전과 후의 성적 차이가 없다.
# H1: 교육 후에 성적이 높아졌다.
from scipy.stats import ttest_rel

before = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15])
after = np.array([10.52, 14.83, 13.03, 16.46, 10.84, 12.45])

help(ttest_rel)
t_stat3, p_value = ttest_rel(after, before, alternative="greater")  # 첫번째 입력값 평균이 더 크다
p_value < 0.05  # True
# p-value가 0.05보다 작으므로 귀무가설을 기각할 수 있다.
# 즉, 교육 후에 성적이 높아졌다.



# 직접 계산하기
# after - before 을 새로운 시리즈로 추가해서 독립 1표본 t 검정을 수행할 수 있다.

# t 검정 통계량을 직접 계산
t_stat3 = (after.mean() - before.mean()) / (after.std(ddof=1) / np.sqrt(len(after)))

# p-value 계산
p_value = t.sf(t_stat3, df=len(after)-1)  # p-value 계산



'''
두 그룹의 분산이 같은지 다른지 체크하는 방법

두 그룹의 분산이 같은지 같지 않은지 검정하기 위해서는 F 검정을 통해 판단할 수 있습니다.
F 검정의 핵심 아이디어는 두 그룹에서 추정한 분산의 비율로 두 그룹의 분산이 같은지 측정합니다.
F 검정은 귀무가설하에서의 검정통계량이 F 분포를 따르기 때문에 붙여진 이름입니다.
'''
# 표본1의 분산 / 표본2의 분산 = F 검정 통계량
# H0: 두 그룹의 분산이 같다.
# H1: 두 그룹의 분산이 다르다.
# F 검정
oj_lengths = np.array([17.6, 9.7, 16.5, 12.0, 21.5, 23.3, 23.6, 26.4, 20.0, 25.2,
25.8, 21.2, 14.5, 27.3, 23.8]) # OJ 그룹 데이터 (15개)

vc_lengths = np.array([7.6, 4.2, 10.0, 11.5, 7.3, 5.8, 14.5, 10.6, 8.2, 9.4,
16.5, 9.7, 8.3, 13.6, 8.2]) 

s1 = oj_lengths.std(ddof=1)
s2 = vc_lengths.std(ddof=1)

f_stat = s1**2 / s2**2
p_val = f.sf(f_stat, dfn=len(oj_lengths)-1, dfd=len(vc_lengths)-1) * 2 # p-value 계산

f.cdf(s2**2/s1**2, 14, 14)  # 왼쪽 p-value 계산
1 - f.cdf(s1**2/s2**2, 14, 14)  # 오른쪽 p-value 계산

# F 검정 통계량과 p-value 계산
from scipy.stats import levene, bartlett
levene(oj_lengths, vc_lengths)  # Levene's test
bartlett(oj_lengths, vc_lengths)  # Bartlett's test



'''
문제 풀이

'''

'''
연습문제 1
표본의 크기가 10이고, 모표준편차를 모르는 상황에서 표본평균이 50, 
표본표준편차가 4일 때, t분포를 이용해  P(T<= 1.812) 를 구하시오.
'''
n = 10
t.cdf(1.812, df=n-1)


'''
연습문제 2
표본의 크기가 15이고, 자유도 14인 
t분포에서 누적확률 0.95에 해당하는 
t값을 구하시오.
'''
n = 14
t.ppf(0.95, df=n)


# 연습문제 3
# 평균이 60이고, 표준편차는 모름. 표본크기 12일 때, 
# P(T>2.18) 일 확률을 구하시오. 

t.sf(2.18, df=11)


# 연습문제 4
# 자동차의 연비는 정규분포를 따른다고 가정한다. 
# 평균 연비가 12km/L이고 표준편차가 1.5km/L인 상황에서 
# 연비가 10km/L 이상일 확률을 파이썬 코드로 계산하시오.
norm.sf(loc=12, scale=1.5, x=10)


# 연습문제 5
# 직원들의 하루 평균 업무시간은 8시간이고, 
# 표준편차는 0.8시간이라고 알려져 있다. 
# 하루 9시간 이상 근무할 확률을 파이썬 코드로 계산하시오. 
# 업무시간은 정규분포를 따른다고 가정한다.
norm.sf(loc=8, x=9, scale=0.8)



'''

연습문제 6
한 프랜차이즈 커피숍은 고객 대기시간을 평균 5분 이하로 유지하려고 합니다. 
50명의 고객을 무작위로 조사한 결과, 대기시간 평균은 5.4분, 
표준편차는 1.2분이었습니다. 대기시간이 평균 5분을 초과하는지 확인하려고 합니다. 
(모표준편차를 안다고 가정)

귀무가설과 대립가설을 설정하시오.
유의수준 5%에서 검정통계량과 p-value를 계산하고 가설을 판단하시오.
'''
# 귀무가설: 대기시간 평균이 5분 이하이다.
# 대립가설: 대기시간 평균이 5분을 초과한다.
z = (5.4 - 5) / (1.2 / np.sqrt(50))  # 검정통계량
p_val = norm.sf(z)  # p-value 계산
p_val < 0.05  # True
# 귀무가설을 기각할 수 있다.
# 즉, 대기시간이 평균 5분을 초과한다고 주장할 수 있다.




'''
연습문제 7
평균 체온은 36.5도라고 알려져 있습니다. 
어떤 실험에서 15명의 체온 측정 결과는 다음과 같습니다:


1. 귀무가설과 대립가설을 설정하시오.
2. 평균 체온이 36.5도와 다른지 검정하시오.
3. 평균 체온에 대한 95% 신뢰구간을 구하시오.
'''


x = np.array([36.3, 36.7, 36.6, 36.5, 36.8, 36.6, 36.4, 36.7, 36.5, 36.3, 36.9, 36.4, 36.2, 36.8, 36.6])
n = len(x)
# 귀무가설: 평균 체온은 36.5도이다.
# 대립가설: 평균 체온은 36.5도가 아니다.

tstat, pval = ttest_1samp(x, popmean=36.5, alternative="two-sided")
pval < 0.05  # False
# p-value가 0.05보다 작지 않으므로 귀무가설을 기각할 수 없다.
# 즉, 평균 체온은 36.5도라고 주장할 수 있다.

critical_val = t.ppf(0.975, df=n-1)  # 95% 신뢰구간의 임계값
# 95% 신뢰구간 계산
lower = x.mean() - critical_val * (x.std(ddof=1) / np.sqrt(n)) 
upper = x.mean() + critical_val * (x.std(ddof=1) / np.sqrt(n))

lower, upper

# 시각화
k = np.linspace(-4, 4, 1000)
plt.plot(k, t.pdf(k, df=n-1), label='t-distribution', color='blue')
plt.axvline(tstat, color='red', linestyle='--', label=f't-statistic = {tstat:.2f}')
plt.fill_between(k, t.pdf(k, df=n-1), where=(k <= -critical_val) | (k >= critical_val), color='red', alpha=0.3, label='Rejection Region')
plt.legend()
plt.xlabel('t-value')
plt.ylabel('Density')
plt.grid()
plt.title("body temperature t-test")
plt.show()


'''
연습문제 8
한 피자 업체의 평균 배달 시간이 30분을 초과하는지 확인하려고 합니다. 
40명의 배달 시간 샘플 평균은 32분, 표준편차는 5분이었습니다.

귀무가설과 대립가설을 설정하시오.
Z-검정을 통해 배달 시간이 평균보다 긴지 확인하시오.
'''
# H0: 평균 배달 시간이 30분 이하이다.
# H1: 평균 배달 시간이 30분을 초과한다.
z = (32 - 30) / 5 / np.sqrt(40)  # 검정통계량
p_val = norm.sf(z)
p_val < 0.05  # False   
# 귀무가설 기각 X 평균 배달 시간이 30분 이하


'''
연습문제 9
고등학생 평균 수면시간이 7시간과 다른지 확인하려고 합니다. 
12명의 샘플 수면시간은 다음과 같습니다:


1. 귀무가설과 대립가설을 설정하시오.
2. 평균 수면시간이 7시간과 다른지 검정하시오.
3. 평균 수면시간의 95% 신뢰구간을 구하시오.
'''
x = np.array([6.5, 6.2, 6.8, 7.1, 6.7, 7.3, 6.9, 7.4, 6.6, 6.8, 7.0, 7.2])
# 귀무가설: 고등학생 평균 수면시간이 7시간이다.
# 대립가설: 고등학생 평균 수면시간이 7시간이 아니다.
tstat, pval = ttest_1samp(x, popmean=7, alternative="two-sided")
p_val < 0.05    # False
# p-value가 0.05보다 작지 않으므로 귀무가설을 기각할 수 없다.
# 즉, 평균 수면시간이 7시간이라고 주장할 수 있다.

critical_value = t.ppf(0.975, df=len(x)-1)

# 95% 신뢰구간 계산
lower_bound = x.mean() - critical_value * (x.std(ddof=1) / np.sqrt(len(x)))
upper_bound = x.mean() + critical_value * (x.std(ddof=1) / np.sqrt(len(x)))
print(lower_bound, upper_bound)


'''
연습문제 10
다이어트 프로그램에 참가한 10명의 참가자들의 체중 변화 전후는 다음과 같습니다:

Before: 75, 72, 78, 80, 69, 77, 73, 76, 74, 71
After: 72, 70, 75, 78, 67, 74, 71, 74, 72, 69
다이어트 프로그램이 체중 감소에 효과가 있는지 확인하려고 합니다.

귀무가설과 대립가설을 설정하시오.
대응표본 t-검정을 통해 평균 체중이 감소했는지 검정하시오.

귀무가설: 체중 변화가 없다.
대립가설: 체중 변화가 있다.
'''
before = np.array([75, 72, 78, 80, 69, 77, 73, 76, 74, 71])
after = np.array([72, 70, 75, 78, 67, 74, 71, 74, 72, 69])
tstat, pval = ttest_rel(after, before, alternative="less")
pval < 0.05  # True
# p-value가 0.05보다 작으므로 귀무가설 기각할 수 있다.
# 즉, 평균 체중이 감소했다고 주장할 수 있다.