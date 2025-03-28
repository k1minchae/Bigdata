from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np

# loc: 구간 시작점, sacle: 구간 길이
rng = np.random.default_rng(seed=2025)
rng.uniform(0, 1)
uniform.pdf(3, loc=2, scale=3)
uniform.pdf(0, loc=2, scale=3)

# 0에서부터 5까지 균일 분포 2, 5 의 pdf 를 그려보세요.
k = np.linspace(0, 5, 100)  # 0~5 까지를 100개로 나눔

# 균등분포의 확률 밀도함수 계산
# loc: 시작값, 끝값: loc + scale

density_k = uniform.pdf(k, loc=2, scale=3)
plt.plot(k, density_k)
plt.xlabel('x')
plt.ylabel('f(x)')


# F(x) = P(X <= x)
# 0에서 7까지 균일분포 (2, 5)의 cdf를 그려보세요.
# 적분해서 넓이로 변환한것
k = np.linspace(0, 7, 100)
cdf_k = uniform.cdf(k, loc=2, scale=3)

plt.plot(k, cdf_k)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

# 균일분포 (2, 5) 표본 5개 추출
uniform.rvs(loc=2, scale=3, size=5)


# 퀀타일 함수 ppf 와 누적 분포함수 cdf 는 역함수 관계
# X~U(1, 2), Fx(1.5) = 1/2
# Fx(A) = 1/2 일때, A는 무엇인지 궁금할 때 사용하는게 ppf (퀀타일 함수)
# 퀀타일 함수: Qx(p) = x


# ppf 함수()
# 균일분포 (2, 5)에서 하위 50%, 40%에 해당하는 값은?
uniform.ppf(0.5, loc=2, scale=3)    # 3.5
uniform.ppf(0.4, loc=2, scale=3)    # 3.4 


# X~U(2, 5) 일 때, P(1 < x <= 4) = ?
uniform.cdf(4, loc=2, scale=3) - uniform.cdf(1, loc=2, scale=3)

# cdf, pdf 에 대해 좀더 살펴보기



# 균일 분포의 기대값 구하는법은?
# x * 확률밀도함수 를 적분
uniform.expect(loc=2, scale=3)


# 균일 분포의 분산은?
uniform.var(loc=2, scale=3)


# 이산형 확률변수의 대표: 베르누이분포
# P(X=k) = p^k * (1-p)^(1-k)
# bernoulli 클래스를 사용해서 베르누이 분포에 대한 함수 사용가능
from scipy.stats import bernoulli
p = 0.3
k = 1   # k는 0또는 1
bernoulli.pmf(k, p)
bernoulli.cdf(k, p)
bernoulli.ppf(k, p)
# bernoulli.rvs(p, size, random_state)


'''
이항 분포 (n, p)
독립적인 베르누이 시행을 n번 반복하여 성공하는 횟수에 대한 분포
베르누이 시행 성공확률: p
공식: nCk * p^k * (1-p)^(n-k)

'''

# X ~ 이항분포 (5, 0.3)
from scipy.stats import binom

# 이 확률 변수의 기대값은?
n, p = 5, 0.3
binom.expect(args=(5, 0.3))
binom(5, 0.3).mean()

# 이 확률 변수의 분산은?
variance = binom.var(n, p)


# 문제: 동전을 7번 던져서 나온 앞면의 수가 3회보다 작은 확률을 구하세요.
# X ~ B(7, 0.47)
# P(X < 3)을 구하면 된다.
p = 0.47    # 앞면이 나올 확률은 0.47
n = 7
sum(binom.pmf(k=i, n=n, p=p) for i in range(3))
binom.cdf(2, n=n, p=p)


# 문제 2-1: 어느 한 공장에서 제품을 하나 만들 때 불량률이 1% 이다.
# 해당 제품은 30개씩 박스에 포장되어 출고된다.
# 박스가 하자가 있다고 판단하는 기준: 3개 이상 불량
# 박스가 불량일 경우의 확률은?
# Y: 박스에 들어있는 불량품 개수
# Y ~ B(n=30, p=0.03)

1 - binom.cdf(2, n=30, p=0.01)



# 문제 2-2
# 회사 A에게 350 박스를 판매함. 
# A는 판매한것의 10% 넘어가는 박스가 불량일 경우 항의전화가 옴
# A회사에게 항의전화 올 확률은?
p_box = 1 - binom.cdf(2, n=30, p=0.01)
1 - binom.cdf(35, n=350, p=p_box) # 0.0 %
1 - binom.cdf(3, n=350, p=p_box) # 3.02 %



# 문제: 포아송분포 확률질량함수 그래프 그려보세요!
# lambda 가 3인 포아송분포의 -1 에서 20까지
from scipy.stats import poisson

poissons = [poisson.pmf(k, 3) for k in range(-1, 21)]
plt.bar(np.arange(-1, 21), poissons)
plt.title('X-Poisson(3)')
plt.xlabel('range of x')
plt.xticks(np.arange(-1, 21, 3))
plt.ylabel('probability')
plt.show()


# X~Poi(3) 에서 표본 5개 뽑는법은?
poisson.rvs(mu=3, size=5)


lambda_hat = np.mean([1, 6, 3, 2, 5, 8, 1, 4, 5])
poisson.pmf(0, mu=lambda_hat)  # 항의 전화 안 올 확률



'''
연습문제 풀이

'''


# 연습문제 1
bernoulli.pmf(0, 0.4)
bernoulli.pmf(1, 0.4)
print(f'1번: {bernoulli.pmf(0, 0.4)}, {bernoulli.pmf(1, 0.4)}')


# 연습문제 2
bernoulli.expect(args=(0.8,))
bernoulli.var(p=0.8)
print(f'2번: {bernoulli.expect(args=(0.8,))}, {bernoulli.var(p=0.8)}')


# 연습문제 3
binom.pmf(k=1, p=0.2, n=3)


# 연습문제 4
1 - binom.cdf(k=4, n=7, p=0.5)


# 연습문제 5
binom.expect(args=(6, 0.3))



# 연습문제 6
poisson.pmf(k=3, mu=2)



# 연습문제 7
1 - poisson.cdf(k=1, mu=4)



# 연습문제 8
poisson.expect(args=(5,))
poisson.var(5)



# 연습문제 9
binom.pmf(k=4, n=10, p=0.6)



# 연습문제 10
poisson.cdf(mu=3.5, k=2)


# 연습문제 11
# Python을 사용하여 베르누이분포 
# 의 확률질량함수를 시각화하는 코드를 작성하시오.

# 연습문제 12
# 베르누이 시행을 4번 반복하여 성공횟수를 세는 이항 확률변수 
# 의 확률분포표를 작성하시오.

# 연습문제 13
# 포아송 분포 
# 에서 
# 을 구하시오.

# 연습문제 14
# 이항 분포 
# 에서 
# 을 누적분포함수의 형태로 계산하시오.

# 연습문제 15
# 다음은 고객이 하루 동안 특정 웹사이트를 방문하는 횟수를 기록한 데이터입니다. 
# 이 데이터가 포아송 분포를 따른다고 가정하고, scipy.stats를 활용하여 다음을 수행하시오.
# 위 데이터를 바탕으로 포아송 분포의 모수를 추정하시오.
# 추정된 람다를 이용하여 포아송 확률질량함수(PMF)를 계산하고, 
# 실제 데이터의 히스토그램과 비교하여 시각화하시오.
# 히스토그램을 겹쳐그릴 것


# 방문 횟수 데이터
visits = np.array([0, 1, 2, 0, 3, 1, 4, 2, 2, 3, 1, 0, 1, 2, 3, 1, 2, 3, 4, 2])

# 관측값
values, counts = np.unique(visits, return_counts=True)
prob_obs = counts / len(visits)

# 추정된 파라미터
lambda_hat = np.mean(visits)
x = np.arange(0, max(values)+1)
pmf_theory = poisson.pmf(x, mu=lambda_hat)

# 시각화
plt.bar(x -   0.2, prob_obs, width=0.4, label="Observed", color="skyblue")
plt.bar(x + 0.2, pmf_theory, width=0.4, label="Poisson Fit", color="orange")
plt.xlabel("Visits")
plt.ylabel("Probability")
plt.title("Observed vs. Fitted Poisson PMF")
plt.legend()
plt.grid(True)
plt.show()