# 카이제곱 검정
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, norm, chi2

# 카이제곱 분포 알아보기
'''
1. 표준정규분포 확률변수를 사용해서 만들 수 있다.
2. n개의 독립적인 표준정규분포 확률변수의 제곱합을 따르는 분포이다.
3. X = Z² ~ Chi-squared(1)이다.
4. X = Z1² + Z2² + ... + Zn² ~ Chi-squared(n)이다.
5. 자유도 n인 카이제곱 분포는 n개의 독립적인 표준정규분포 확률변수의 제곱합을 따르는 분포이다.
6. 카이제곱 분포는 비대칭적이고, 오른쪽으로 긴 꼬리를 가진 분포이다.
7. 카이제곱 분포는 n이 커질수록 정규분포에 가까워진다.

'''

# 자유도에 따른 카이제곱 분포 시각화
k = np.linspace(0, 40, 200)
pdf_chi = chi2.pdf(k, df=3)
pdf_chi_7 = chi2.pdf(k, df=20)
plt.plot(k, pdf_chi, label='Chi-squared(df=3)', color='red')
plt.plot(k, pdf_chi_7, label='Chi-squared(df=20)', color='green')
plt.title('Chi-squared Distribution with Different Degrees of Freedom')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()



# 표준정규분포에서 3개 샘플 → 제곱합
z = norm.rvs(size=3 * 10000)
z = np.reshape(z, (10000, 3))
def sum_of_squares(x):
    return np.sum(x ** 2)
chi_samples = np.apply_along_axis(sum_of_squares, axis=1, arr=z)
# 카이제곱분포 (df=3)
k = np.linspace(0, 20, 200)
pdf_chi = chi2.pdf(k, df=3)
plt.hist(chi_samples, bins=30, density=True, edgecolor="black", alpha=0.5)
plt.plot(k, pdf_chi, label='Chi-squared(df=3)', color='green')
plt.title('Sum of squares of 3 N(0,1) vs Chi-squared(df=3)')
plt.legend()
plt.show()



# 이항분포 근사 시각화
# 이항분포는 n개의 독립적인 베르누이 시행의 성공 횟수를 나타내는 분포이다.
# n*p >= 5, n*(1-p) >= 5일 때 정규분포로 근사할 수 있다.

n, p = 100, 0.5
sample = binom.rvs(n=n, p=p, size=10000)

# 정규 근사값 (mean=np, std=sqrt(np(1-p)))
k = np.linspace(min(sample), max(sample), 100)
pdf_norm = norm.pdf(k, loc=n*p, scale=np.sqrt(n*p*(1-p)))
plt.hist(sample, bins=30, density=True, edgecolor="black", alpha=0.5)
plt.plot(k, pdf_norm, label='Normal Approx', color='red')
plt.title(f'Binomial(n={n}, p=0.5) vs Normal Approximation')
plt.legend()
plt.show()


# 실습
# X ~ B(n=100, p=0.2)
sigma = np.sqrt(100 * 0.2 * (1 - 0.2))
norm.cdf(loc=20, scale=sigma, x=40)  # 0.9999997
binom.cdf(40, n=100, p=0.2) # 0.9999987

binom.cdf(15, n=100, p=0.2)  # 0.1285
norm.cdf(loc=20, scale=sigma, x=15)  # 01056

'''
예제

베어링 제조 회사의 품질 관리를 맡고 있는 정부 기관의 규정에 따르면, 생산되는 제품의 금속 재질
함유량 분산이 1.3 을 넘으면 생산 부적격이라고 판단한다. 다음은 A 회사 제품의 금속 함유량를 검
사한 데이터이다. 데이터를 기준으로 해당회사의 생산 부적격 검정을 수행하시오. 단, 유의 수준은
5%로 설정하시오.
'''
x = np.array([10.67, 9.92, 9.62, 9.53, 9.14, 9.74, 8.45, 12.65, 11.47, 8.62])
# H0: 분산이 1.3 이하이다.
# H1: 분산이 1.3 초과이다.

n = len(x)
s2 = x.var(ddof=1)

t = (n - 1) * s2 / 1.3  # 카이제곱 검정 통계량
pval = 1 - chi2.cdf(t, df=n-1)
pval < 0.05  # False
# p-value가 0.05보다 크므로 귀무가설을 채택한다.
# 데이터의 분산이 1.3 이하이다.

# 만약, 해당 데이터로 95% 신뢰구간을 구하고 싶다면?
# 카이제곱 분포의 분위수를 사용해서 구할 수 있다.
# P(a < (n-1) * s2 / sigma² < b) = 0.95
v1 = chi2.ppf(0.025, df=n-1)
v2 = chi2.ppf(0.975, df=n-1)

lower = (n - 1) * s2 / v1
upper = (n - 1) * s2 / v2



1/3.543, 1/0.5038


'''
독립성 검정

둘다 범주형 변수일 때 사용합니다.
독립성 검정은 주어진 표에서 두 카테고리 변수 간의 상관성이 있는지 여부를 분석하는 방법입니다.
이 검정을 통해 두 변수가 독립적인지 아닌지를 판단할 수 있습니다.

H0: 두 변수는 독립적이다.
H1: 두 변수는 독립적이지 않다.

카이제곱 검정은 두 변수의 독립성을 검정하는 방법으로, 관측된 빈도와 기대 빈도의 차이를 측정합니다.

카이제곱 통계량은 다음과 같이 계산됩니다.
X² = Σ((O - E)² / E)
O: 관측 빈도 (Observed frequency)
E: 기대 빈도 (Expected frequency)
'''

# 예제
# 운동 선수 18명, 일반인 10명에 대한 흡연 여부 조사 데이터
# 운동선수: 비흡연자 14, 흡연자 4
# 일반인: 비흡연자 0, 흡연자 10
# (원래꺼 - 예상치) 제곱 / 예상치 의 합


from scipy.stats import chi2_contingency

table = np.array([[14, 4],
                [0, 10]])

# correlation = False 로 해야지 근사치로 계산 안해줌.
chi2, p, df, expected = chi2_contingency(table, correction=False)
print('X-squared:', chi2.round(3), 'df:', df, 'p-value:', p.round(3))
# p-value가 0.05보다 작으므로 귀무가설을 기각한다.
# 두 변수는 독립적이지 않다.





############## 4/3 수업 내용 ################


# 1. 적합도 검정 (Goodness of Fit Test)
# 목적: 한 집단이 어떤 이론적 분포(기대비율) 를 따르는지 검정
# 예시: 주사위를 60번 던졌을 때, 각 눈(1~6)이 동일한 확률(1/6) 로 나왔는지 확인

# 귀무가설: 주사위는 공정하다 (각 눈이 동일한 확률로 나왔다)
# 대립가설: 주사위는 공정하지 않다 (각 눈이 동일한 확률로 나오지 않았다)


# 2. 동질성 검정 (Test of Homogeneity) => 같은 분포를 따르는지
# 목적: 두 개 이상의 집단이 같은 분포(성향) 를 가지는지 검정
# 예시: A학교, B학교, C학교 학생들이 좋아하는 과목 비율이 동일한지 검정


# 3. 독립성 검정 (Test of Independence) => 관계성이 있는지
# 목적: 두 범주형 변수 간의 관련성(독립성) 을 검정
#       즉, A와 B라는 두 변수 간에 관계가 있는가?
# 예시: 성별과 선호하는 과목 간의 관련성 검정   
# 수집 방식: 교차표 형태


# ✅ 동질성 검정:
# “두 집단 이상이 같은 분포를 따르냐?”
# ✔️ 같은 분포면 OK! 관계가 있는지는 상관 없음
# 📌 예: 남학생과 여학생이 과목 선호도 분포가 같은지 비교

# ✅ 독립성 검정:
# “두 변수 간에 관련이 있냐? (독립이냐? 아니냐?)”
# ✔️ 성별에 따라 과목 선호도가 달라지면 독립이 아님(관계 있음)
# 📌 예: 성별과 과목 선호도가 서로 관련이 있는지 확인


# 동질성 검정 하기
from scipy.stats import chisquare

# 관측된 데이터 (Observed data)
x = np.array([13, 23, 24, 20, 27, 18, 15])  # 각 범주에서 관측된 빈도

# 기대 빈도 (Expected data)
# 모든 범주에서 동일한 빈도를 기대한다고 가정 (예: 20으로 동일하게 설정)
exp_x = np.repeat(20, 7)  # [20, 20, 20, 20, 20, 20, 20]

# 카이제곱 통계량 계산 (수식 기반)
# 카이제곱 통계량 = Σ((관측값 - 기대값)^2 / 기대값)
chi_square_stat = sum((x - exp_x) ** 2 / exp_x)  # 7.6
print(f"카이제곱 통계량 (수식 기반): {chi_square_stat}")

# scipy를 사용하여 카이제곱 검정 수행
# chisquare 함수는 관측값과 기대값을 입력받아 검정 통계량과 p-value를 반환
statistic, pval = chisquare(x, f_exp=exp_x)
print(f"검정 통계량: {statistic:.2f}, p-value: {pval:.2f}")

# 유의 수준 0.05와 비교하여 결과 해석
# p-value가 0.05보다 크므로 귀무가설을 기각하지 않는다.
# 귀무가설: 두 집단은 같은 분포를 따른다.
if pval < 0.05:
    print("귀무가설을 기각합니다. 두 집단은 같은 분포를 따르지 않습니다.")
else:
    print("귀무가설을 기각하지 않습니다. 두 집단은 같은 분포를 따릅니다.")


