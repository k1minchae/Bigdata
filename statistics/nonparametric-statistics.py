# 비모수검정
import numpy as np
from scipy.stats import rankdata, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns


'''
비모수검정의 장점
1. 분포가정이 없다
2. 이상치에 민감하지 않다.
3. 통계량이 직관적인 경우가 많다.
4. 데이터의 크기에 구애받지 않는다.
'''

# 비모수 검정을 매번 사용하지 않는 이유
# 모수검정이 비모수 검정보다 검정력이 더 높다.

'''
비모수 검정에 대한 기초 지식들
1. 비모수 검정은 모수의 중심점을 평균이 아닌 중앙값으로 설정함.
2. 중앙값에 대해 검정한다.
3. 그래서, 모 중앙값을 나타내는 그리스 문자(𝜂) 에타를 사용한다.
'''


# 귀무가설(H0): 𝜂 = 𝜂0 
# 대립가설(H1): 𝜂 ≠ 𝜂0

# 앞에서 배웠던 t 검정의 귀무가설과 대립가설 형태와 비슷하지만, 
# 모평균이 아닌 모중앙값에 대한 검정이라는 것만 다릅니다.


# 검정 통계량과 핵심 아이디어
# 윌콕슨 순위합 검정 통계량(Wilcoxon rank-sum test statistic)
# 1. 두 집단의 데이터를 합친다.
# 2. 두 집단을 합친 데이터에 대해 순위를 매긴다.
# 3. 두 집단의 데이터에 대해 각각 순위를 매긴다.
# 4. 두 집단의 순위합을 계산한다.
# 5. 순위합을 기준으로 검정 통계량을 계산한다.
# 6. 검정 통계량을 기준으로 p-value를 계산한다.
# 7. p-value를 기준으로 귀무가설을 기각할지 여부를 결정한다.


# 문제 1
# Ri 들을 모두 더하면 ? 
# n(n+1)/2 가 된다.

# 윌콕스 순위합 검정 통계량 결과 해석


# 예시 (1표본 검정)
# 1표본 검정은 단일 집단의 중앙값이 특정 값과 같은지를 검정하는 방법입니다.
'''
$$
\psi_i = \psi(x_i - \eta_0) =
\begin{cases}
1 & \text{if } x_i - \eta_0 > 0 \\
0 & \text{otherwise}
\end{cases}
$$
'''
sample = np.array([9.76, 11.1, 10.7, 10.72, 11.8, 6.15, 10.52,
14.83, 13.03, 16.46, 10.84, 12.45])

eta_0 = 10
len(sample)

# Ri를 계산해보자
# eta_0에서 떨어진 거리의 순위를 계산한다.
ri = rankdata(np.abs(sample - eta_0))
psi_i = np.where(sample - eta_0 > 0, 1, 0)

sum(ri)  # 78.0
w_plus = sum(psi_i * ri)
w_minus = sum((1 - psi_i) * ri) # sum(ri) - w_plus
print(w_plus, w_minus)  # 67.0 11.0

# wilcoxon 검정 통계량을 계산해보자
stat, pval = wilcoxon(sample - eta_0, alternative='two-sided')
stat, pval  # 11.0  0.0268
# 기각 가능


# 
# 예시 (2표본 검정)
# 2표본 검정은 두 집단의 중앙값이 같은지를 검정하는 방법입니다.

# U = min(U1, U2)
# 1번 그룹: x11, x12, x13  5, 1, 3  (n1 = 3)
# 2번 그룹: x21, x22       7, 9     (n2 = 2)
# R1: x12, x13, x11, x21, x22
# eta1이 eta2보다 작아서, 1번 그룹이 전부 왼쪽에 있다.

#  등분산 가정이 깨졌을 때 2표본 검정
from scipy.stats import brunnermunzel



# 윌콕슨 순위합 검정은 1표본 t검정과 비슷하다.
# (with 레빈검정)맨휘트니 U 검정은 2표본 t검정(with F검정)과 비슷하다.
# (with 레빈검정)BM검정은 2표본 t검정 (등분산 X) 과 비슷하다.


# 대응표본 비모수 검정
# 대응표본은 2개 그룹이 있는 데이터처럼 보이지만, 실제로는 1개 그룹의 데이터입니다.




'''

연습문제

'''
import pandas as pd
import numpy as np

# 1번
brand_a = [35.2, 35.5, 35.6, 35.7, 35.8]
brand_b = [36.2, 37.0, 38.2, 38.0, 39.5]
data = pd.DataFrame({
    'Product': ['A'] * 5 + ['B'] * 5,
    'Quantity': brand_a + brand_b
})

# 두 제품 브랜드 간 포장 단위당 실제 중량 차이가 있는지 알아보기 위해 
# 비모수 검정을 실시하시오.

from scipy.stats import levene, mannwhitneyu
stat, pval = levene(brand_a, brand_b)
pval < 0.05
# 동분산성 만족 => 독립 2표본 t검정 => 맨휘트니 U 검정

stat, p = mannwhitneyu(brand_a, brand_b, alternative='two-sided')
p < 0.05 # True
# 귀무가설 기각: 두 브랜드 간 실제 중량차이가 있다.





# 2번
# 데이터 설명

# Trained : 훈련을 받은 그룹의 시험 점수
# Untrained : 훈련을 받지 않은 그룹의 시험 점수 (같은 쌍에서 추출)
trained = np.array([115, 123, 137, 140, 130])
untrained = np.array([113, 119, 130, 134, 125])

from scipy.stats import wilcoxon

stat, pval = wilcoxon(trained, untrained)
pval < 0.05
# 귀무가설 기각 X
# 훈련을 받은 그룹과 받지 않은 그룹과 차이가 없다.




# 3번

# 데이터 설명

# Group : 평가 그룹 (전문가, 초급자, 학생)
# Accuracy : 테스트 정답 수

data = {
    'Experts': [82, 80, 85, 83, 88],
    'Novices': [77, 70, 74, 79, 73],
    'Students': [68, 72, 75, 73, 70]
}
df = pd.DataFrame(data)

from scipy.stats import kruskal
stat, pval = kruskal(df['Experts'], df['Novices'], df['Students'])
pval < 0.05 # True
# 세 집단 간 차이 O




# 4번
# 데이터 설명

# DrugA, DrugB : 두 가지 약물을 복용한 후 반응 시간 (초)
drugA = [1.82, 2.12, 1.74, 2.10, 1.65, 1.91, 2.45]
drugB = [2.22, 2.48, 2.15, 2.68, 2.60, 4.65]
stat, pval = levene(drugA, drugB)
pval < 0.05 # 등분산성 만족

stat, pval = mannwhitneyu(drugA, drugB, alternative='two-sided')
pval < 0.05     # True
# 두 약물간 반응 시간 차이가 유의미하다.