import numpy as np


'''
문제 1. 부모의 심장 질환 이력과 남성의 사망 원인 간 상관관계 분석
1999년, 한 대학병원의 공중 보건 연구원이 발표한 연구 결과입니다.

사망한 937명의 남성들의 의료 기록을 조사한 결과, 
이들 중 210명이 당뇨병 관련 합병증으로 사망한 것을 발견했습니다. 
또한, 937명 중 312명은 최소한 한 명의 부모가 당뇨병을 겪었으며, 
이 중 102명은 당뇨병 관련 합병증으로 사망했습니다.
이 데이터를 바탕으로, 부모가 모두 당뇨병을 겪지 않은 남성이 
당뇨병 관련 합병증으로 사망할 확률을 계산하세요.(소수점 넷째 자리에서 반올림하세요.)
'''
# 전체 조사 대상 남성: 937명
# 당뇨병 관련 합병증으로 사망한 남성: 210명
# 부모 중 최소 한 명이 당뇨병을 겪은 남성: 312명
# 이 중 당뇨병 관련 합병증으로 사망한 남성: 102명


# 부모가 당뇨병을 겪지 않은 남성: 937−312=625 (분모)
# 부모가 당뇨병을 겪지 않은 남성 중 당뇨병 관련 합병증으로 사망한 남성: 210-102=108 (분자)

# 부모님이 모두 당뇨병 X 당뇨 합병증 사망
ans1 = (210 - 102) / (937 - 312)
ans1 = round(ans1, 3)



'''
문제 2. 대학 도서관 및 체육관 이용 학생의 방문 확률 분석
어떤 대학교에 재학 중인 학생들 중에서 22%는 도서관과 체육관을 모두 이용하는 것으로 나타났으며,
12%는 이들 중 어느 곳도 이용하지 않습니다. 
학생이 체육관을 이용할 확률은 도서관을 이용할 확률보다 0.14만큼 더 높습니다.

이 학생들 중에서 무작위로 선택된 한 학생이 도서관을 이용할 확률을 계산하세요.
'''
# 도서관, 체육관 모두 이용: 0.22
# 아무곳도 이용 X: 0.12
# 도서관 이용 = a
# 체육관 이용 = a + 0.14
# 1 = a + a + 0.14 - 0.22 + 0.12
# 2a = 1 - 0.14 + 0.22 - 0.12
a = (1 - 0.14 + 0.22 - 0.12) / 2
ans2 = round(a, 2)




'''
문제 3. 도서관 이용 패턴 분석
한 학교에서 학생들의 도서관 이용 패턴을 조사했습니다. 
이 연구에서 다음과 같은 결론을 도출했습니다.

학생들은 고전문학을 빌리는 확률보다 소설을 빌리는 확률이 두 배 높습니다.
학생이 고전문학을 빌리는 사건과 소설을 빌리는 사건은 서로 독립적입니다.
학생이 고전문학과 소설을 동시에 빌릴 확률은 0.08입니다.
이 정보를 바탕으로, 학생이 도서관에서 고전문학도 소설도 빌리지 않을 확률을 계산하세요.
'''
# 소설 빌리는 확률: x
# 고전 문학 빌리는 확률: 2x
# x^2 = 0.04
# x = 0.2
# 1 = (0.2 + 0.4) - 0.08 + answer
answer = 1 + 0.08 - 0.2 - 0.4
ans3 = round(answer, 2)




'''
문제 4. 기업 직원의 건강 보험 선택 분석
대기업의 보험사는 직원들에게 선택적 건강 보험 패키지를 제공합니다. 
이 보험 패키지는 A, B, C 세 가지 보험 옵션이 있으며, 
직원들은 이 중에서 두 가지를 선택하거나 선택하지 않을 수 있습니다. 
보험 A와 B만을 선택한 직원의 비율은 1/12, 
A와 C만을 선택한 비율은 1/6, 
B와 C만을 선택한 비율은 1/4입니다.

이 정보를 사용하여 임의의 직원이 아무런 보험도 선택하지 않았을 확률을 계산하세요.
'''
# a & b = 1/12
# a & c = 1/6
# b & c = 1/4
ans4 = 1 - (1/12 + 1/6 + 1/4)


'''
문제 5. 음식점 체인의 지출 보고서에서 노동 비용 포함 확률 분석
음식점 체인은 각 지점에서 발생하는 지출 보고서 중 
85%가 식재료 비용 또는 노동 비용을 포함하고 있습니다. 
지출 보고서 중 식재료 비용이 포함되지 않는 보고서의 비율은 전체의 25%입니다.

식재료 비용의 발생이 노동 비용의 발생과 독립적이라고 가정할 때, 
지출 보고서에 노동 비용이 포함될 확률을 계산하시오.
'''
# P(식재료 ∪ 노동) = 0.85
# P(식재료) = 0.75
# P(식재료 ∩ 노동) = P(식재료) + P(노동) - P(식재료 ∪ 노동)
# P(노동) = x
# 0.75 + x - 0.85 = P(식재료 ∩ 노동)
# P(식재료 ∩ 노동) = P(식재료) * P(노동)
# 0.75 * x = 0.75 + x - 0.85
# 0.75x - x = -0.1
# -0.25x = -0.1
# x = 0.4
ans5 = 0.4


'''
문제 6. 음식점 체인의 지출 보고서에서 노동 비용 포함 확률 분석
한 건강 보험 회사는 모든 연령대의 가입자를 보장합니다. 
어느 보험계리사가 보험에 가입한 사람들에 대한 통계를 다음과 같이 정리했습니다.

연령대	질병 발생 확률	보험 가입자 비율
16-20	0.06	0.08
21-30	0.03	0.15
31-65	0.02	0.49
66-99	0.04	0.28
한 무작위로 선택된 보험 가입자가 질병에 걸렸다는 조건 하에, 
그 사람이 16-20세 연령대일 확률을 구하세요.(소수 셋째 자리에서 반올림 하세요.)
'''
# 보험 가입자가 질병에 걸릴 확률
pain = 0.06 * 0.08 + 0.03 * 0.15 + 0.02 * 0.49 + 0.04 * 0.28
# 16~20 대가 질병에 걸릴 확률
pain_16 = 0.08 * 0.06
ans6 = round(pain_16 / pain, 3)


'''
문제 7. 판매량 X의 평균과 표준편차 내 비율 계산
다음은 온라인 쇼핑몰에서 판매되는 특정 제품의 일일 판매량에 대한 확률 분포입니다.

판매량(개)	확률
5	0.10
10	0.15
15	0.20
20	0.30
25	0.15
30	0.05
35	0.05
평균 판매량의 1 표준편차 이내에 속하는 판매량의 비율을 계산하십시오.
'''
# 평균 계산
x = np.arange(5, 36, 5)
p_x = np.array([0.1, 0.15, 0.2, 0.3, 0.15, 0.05, 0.05])
e_x = np.sum(x * p_x)

# 분산 계산
var_x = np.sum(((x - e_x)**2) * p_x)

# 표준편차 계산
std_x = np.sqrt(var_x)

# 1 표준편차 범위
lower_bound = e_x - std_x
upper_bound = e_x + std_x

# 1 표준편차 내의 판매량 비율 계산
within_std = np.sum(p_x[(x >= lower_bound) & (x <= upper_bound)])
ans7 = round(within_std, 2)


'''
문제 8. 상금 지급 기금의 최대 지급액 분석
어떤 학교는 특정한 성취를 달성한 학생에게 상금을 지급하기 위해 120만원의 기금을 마련했습니다. 
이 학교의 학생 20명 중 각 학생이 다음 해에 성취를 달성할 확률은 2%입니다. 
서로 다른 학생이 성취를 달성하는 사건은 상호 독립적입니다.

기금이 모든 성취에 대한 지급을 충당하지 못할 확률이 1% 미만이 되도록 하는 
상금의 최대 값을 계산하세요.
'''
from math import comb

# 주어진 조건
n = 20  # 학생 수
p = 0.02  # 각 학생이 성취할 확률
fund = 1_200_000  # 총 기금

import matplotlib.pyplot as plt
# 이항 분포 확률 계산 함수
def func(size):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(size, 21))

# 이항 분포 그래프 그리기 (x를 정수 단위로 표시)
x_values = range(0, n + 1)
y_values = [func(i) for i in range(n+1)]
y_values = [sum(binom.pmf(k=i, n=20, p=0.02) for i in range(size, 20)) for size in range(n + 1)]

plt.plot(x_values, y_values, marker='o', color='blue', linestyle='-', linewidth=2, markersize=5)
plt.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, label='y=0.01')
plt.title('n=20, p=0.02')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.ylim((0.0, 0.1))
plt.xlim((0, 5))
plt.legend()
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.show()


from scipy.stats import binom
# Y ~ B(20, 0, 0.02)
# P(Y = 0) = 20C0 * 0.09 ** 20 * 0.02 ** 0
sum(binom.pmf(k=i, n=20, p=0.02) for i in range(2, 20))
