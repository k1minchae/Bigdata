import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform, norm, t, binom

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
