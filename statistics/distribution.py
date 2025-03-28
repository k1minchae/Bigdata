from scipy.stats import norm, uniform, poisson, binom, bernoulli
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 정규분포

'''
빅분기 2024년 전체 평균 30점, 표준편차 5 (모분포)
LS 빅데이터 스쿨 20명 학생이 (표준분포) 빅분기를 친 
성적 평균이 32점보다 클 확률은?
'''

x = norm(loc=30, scale=np.sqrt(5**2 / 20))
1 - x.cdf(32)



'''
X~U(2, 14) 를 따를 때, X에서 30개 표본을 뽑은 표본 평균이 6보다 작을 확률은?
'''
# 균일분포의 평균과 분산 계산
mean = uniform(loc=2, scale=12).mean()  # loc=2, scale=14-2=12
variance = uniform(loc=2, scale=12).var()

# 표본 평균의 분포: 정규분포로 근사
x = norm(loc=mean, scale=np.sqrt(variance / 30))

# 표본 평균이 6보다 작을 확률 계산
x.cdf(6)



# 연습 문제

# 1. 지수분포
from scipy.stats import expon

# 지수분포는 scale 에 역수넣어야함
x = 2
lambda_ = 0.5
prob = expon.cdf(x, scale=1/lambda_)  # 따라서 scale은 2로 설정됩니다.

# 1-2. 지수분포
lambda_ = 2
1 - expon.cdf(1, scale=1/lambda_)

# 1-3. 지수분포
# 지수분포의 기대값은 1/lambda_, 분산은 1/lambda_^2
lambda_ = 3
expon.mean(scale=1/lambda_)  # 기대값 (mean)
expon.var(scale=1/lambda_)  # 분산 (variance)


# 2. 균일분포
u = uniform(loc=2, scale=3)
u.cdf(4) - u.cdf(3)


# 2-1. 균일분포
u = uniform(loc=0, scale=8)
u.mean()
u.var()


# 고장 간격이 평균 10시간인 전자부품이 있다. 
# 이 부품의 고장시간은 지수분포를 따른다고 가정할 때, 5시간 이상 동작할 확률을 구하시오.
lambda_ = 1/10
1 - expon.cdf(5, scale=1/lambda_)



# 어떤 콜센터에 전화가 평균 6분 간격으로 걸려온다. 
# 3분 이내에 전화가 걸려올 확률을 구하시오.
lambda_ = 1/6
expon.cdf(3, scale=1/lambda_)


# 한 ATM 기계에 고객이 평균적으로 12분에 한 번씩 도착한다고 할 때, 
# 다음 고객이 10분 이상 기다릴 확률을 구하시오.

# 평균 도착 간격이 12분이므로, lambda는 1/12
lambda_ = 1 / 12

# 10분 이상 기다릴 확률은 1 - CDF(10)
1 - expon.cdf(10, scale=1/lambda_)
from scipy



# 어떤 고속도로에서 차량이 평균 5초 간격으로 진입한다고 할 때, 
# 특정 차량이 2초 이내에 들어올 확률을 구하시오.
lambda_ = 1/5
expon.cdf(2, scale=1/lambda_)


# 지수분포 람다가 0.5일 때 P(X<=3) 을 계산하세요.
expon.cdf(3, scale=1/0.5)



# X값 생성
x = np.linspace(0, 7, 100)

# 람다가 2일 때 지수분포의 누적분포함수
y = expon.cdf(x, scale=1/2)

# 시각화
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=f'Exponential CDF (λ={2})', color='blue')
plt.title('Exponential Distribution CDF')
plt.xlabel('x')
plt.ylabel('CDF')
plt.legend()
plt.grid()
plt.show()



# U(2, 6) 의 PDF 그리기
x = np.linspace(2, 6, 1000)  # 2부터 6까지 100개의 점 생성
y = uniform.pdf(x, loc=2, scale=4)  # loc=2, scale=6-2=4
plt.figure(figsize=(8, 5))
plt.bar(x, y, label='U(2, 6)', color='skyblue', width=0.01)
plt.title('Uniform Distribution PDF')
plt.xlabel('x')
plt.ylabel('PDF')
plt.ylim((0, 0.5))
plt.legend()
plt.grid()
plt.show()



# U(0, 10) 일 때 P(X>7)
1 - uniform.cdf(7, loc=0, scale=10)



# Python을 사용하여 지수분포(평균 4시간)를 따르는 기계의 
# 고장 간격이 6시간 이상일 확률을 구하시오.
lambda_ = 1/4
1 - expon.cdf(6, scale=1/lambda_)




# 포아송분포 예제 문제

# 1. 평균적으로 3분에 2번의 전화가 걸려오는 콜센터에서, 
# 5분 동안 4번의 전화가 걸려올 확률은?
mu = (2 / 3) * 5  # 5분 동안의 평균 발생 횟수
poisson.pmf(k=4, mu=mu)  # 정확히 4번 발생할 확률


# 2. 어떤 웹사이트에 평균적으로 10분에 3번의 방문이 발생한다고 할 때, 
# 10분 동안 방문이 5번 이상 발생할 확률은?
mu = 3
1 - poisson.cdf(k=4, mu=mu)


# 3. 한 공장에서 평균적으로 1시간에 6개의 불량품이 발생한다고 할 때, 
# 1시간 동안 불량품이 2개 이하로 발생할 확률은?
mu = 6
poisson.cdf(k=2, mu=6)


# 4. 평균적으로 하루에 8건의 사고가 발생하는 도시에서, 
# 하루에 사고가 10건 발생할 확률은?
poisson.pmf(k=10, mu=8)


# 5. 평균적으로 1분에 4개의 이메일이 도착하는 메일 서버에서, 
# 30초 동안 이메일이 3개 도착할 확률은?
mu = 4 * (30 / 60)
poisson.pmf(k=3, mu=mu)


# 정규분포 문제 
# 1. 한 도시의 평균 기온이 20도, 표준편차가 3도인 정규분포를 따를 때, 
#    특정 날의 기온이 25도보다 높을 확률은?
1 - norm.cdf(25, loc=20, scale=3)


# 2. 한 회사의 직원 평균 연봉이 5천만 원, 표준편차가 1천만 원인 정규분포를 따를 때, 
#    직원 연봉이 4천만 원에서 6천만 원 사이에 있을 확률은?
n = norm(loc=5000, scale=1000)
n.cdf(6000) - n.cdf(4000)


# 3. 한 학교의 학생 평균 시험 점수가 75점, 표준편차가 10점인 정규분포를 따를 때, 
#    특정 학생의 점수가 60점보다 낮을 확률은?
n = norm(loc=75, scale=10)
n.cdf(60)

# 4. 한 공장의 제품 무게가 평균 500g, 표준편차가 20g인 정규분포를 따를 때, 
#    제품 무게가 480g에서 520g 사이에 있을 확률은?
n = norm(loc=500, scale=20)
n.cdf(520) - n.cdf(480)



# 5. 한 대학의 학생 평균 공부 시간이 하루 6시간, 표준편차가 1.5시간인 정규분포를 따를 때, 
#    특정 학생이 하루 8시간 이상 공부할 확률은?
n = norm(loc=6, scale=1.5)
1 - n.cdf(8)



# 지수분포 문제 
# 1. 평균 4시간 간격으로 고장나는 기계가 3시간 이내에 고장날 확률은?
lambda_ = 1/4
expon.cdf(3, scale=1/lambda_)

# 2. 평균 10분 간격으로 전화가 오는 콜센터에서 5분 이내에 전화가 올 확률은?
lambda_ = 1/10
expon.cdf(5, scale=1/lambda_)

# 3. 평균 6초 간격으로 차량이 진입하는 고속도로에서 8초 이상 차량이 진입하지 않을 확률은?
lambda_ = 1/6
1 - expon.cdf(8, scale=1/lambda_)

# 4. 평균 2시간 간격으로 이벤트가 발생하는 시스템에서 1시간 이내에 이벤트가 발생할 확률은?
lambda_ = 1/2
expon.cdf(1, scale=1/lambda_)

# 5. 평균 15분 간격으로 고객이 도착하는 매장에서 20분 이상 고객이 도착하지 않을 확률은?
lambda_ = 1/15
1 - expon.cdf(20, scale=1/lambda_)


# 균일분포 문제 
# 1. 0분에서 10분 사이에 도착하는 버스가 3분에서 7분 사이에 도착할 확률은?
uniform(loc=0, scale=10).cdf(7) - uniform(loc=0, scale=10).cdf(3)

# 2. 5km에서 15km 사이를 이동하는 택시가 10km 이하를 이동할 확률은?
uniform(loc=5, scale=10).cdf(10)

# 3. 2시간에서 8시간 사이에 완료되는 작업이 4시간에서 6시간 사이에 완료될 확률은?
uniform(loc=2, scale=6).cdf(6) - uniform(loc=2, scale=6).cdf(4)

# 4. 1kg에서 5kg 사이의 무게를 가진 물건이 2kg 이상일 확률은?
1 - uniform(loc=1, scale=4).cdf(2)

# 5. 0도에서 20도 사이의 온도를 가진 날씨가 15도 이하일 확률은?
uniform(loc=0, scale=20).cdf(15)


# 이항분포 문제 
# 1. 성공 확률이 30%인 제품 검사를 10번 했을 때, 4개의 불량품이 나올 확률은?
binom.pmf(p=0.3, n=10, k=4)

# 2. 성공 확률이 60%인 설문조사에서 15명을 조사했을 때, 10명 이상이 긍정적으로 응답할 확률은?
1 - binom.cdf(p=0.6, k=9, n=15)

# 3. 성공 확률이 50%인 동전을 8번 던졌을 때, 앞면이 3번 이하로 나올 확률은?
binom.cdf(p=0.5, k=3, n=8)

# 4. 성공 확률이 40%인 시험 문제를 12문제 풀었을 때, 5문제를 맞출 확률은?
binom.pmf(p=0.4, n=12, k=5)

# 5. 성공 확률이 70%인 고객 응대에서 20명의 고객 중 15명 이상이 만족할 확률은?
1 - binom.cdf(p=0.7, n=20, k=14)




# 학교에서는 특정 성취를 달성한 학생에게 상금을 지급합니다.



# 하지만 준비된 기금(1,200,000원)이 모든 학생에게 지급되도록 보장해야 합니다.

# 학생 수: 
# n=20
# 성취할 확률: p=0.02 (각 학생이 성취할 확률)
# 기금: 1,200,000원
# 목표: 지급해야 하는 상금의 최대값을 찾아야 함.
# 조건: 기금이 부족할 확률이 1% 미만이어야 함.

binom.pmf(n=20, k=1, p=0.02)
binom.pmf(n=20, k=2, p=0.02)
binom.pmf(n=20, k=3, p=0.02)