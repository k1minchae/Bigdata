# 사후검정
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

'''
분산분석은 3개 이상의 집단 간 평균 차이가 
통계적으로 유의미한지를 검정하는 데 사용됩니다. 
그러나 분산분석 자체는 어느 그룹 간에 차이가 있는지는 알려주지 않습니다.

사후 검정은 분산 분석 후, 검정 결과가 유의미하게 나왔을 경우, 세부적으로 어느 그룹간 평균 차이
가 통계적으로 유의미한지 추가 검정을 할 때 사용됩니다. 이를 구체적으로 알아보기 위해 사후 검정
(post-hoc tests)이 필요합니다. 사후 검정은 다중 비교 문제를 다루며, 이를 통해 구체적인 그룹 간
차이를 확인할 수 있습니다.

[사후검정의 필요성]
- 1종 오류(유의미한 차이가 없는데 유의미하다고 판단하는 오류)를 줄이기 위해
왜냐하면, 다중 비교를 할 때, 각 비교마다 1종 오류가 발생할 확률이 증가하기 때문입니다.

예를 들어, 3개의 집단 A, B, C가 있을 때, A와 B, A와 C, B와 C를 비교하면 총 3번의 비교가 이루어지며,
각 비교마다 1종 오류가 발생할 확률이 증가합니다.
따라서, 사후 검정을 통해 다중 비교 문제를 해결하고, 각 그룹 간의 차이를 명확히 확인할 수 있습니다.

[참고]
검정력이란?
- 검정력은 귀무가설이 거짓일 때, 이를 올바르게 기각할 확률을 의미합니다.
제 1종 오류란?
- 귀무가설이 참일 때, 이를 잘못 기각하는 오류를 의미합니다.
제 2종 오류란?
- 귀무가설이 거짓일 때, 이를 잘못 채택하는 오류를 의미합니다.


[사후검정 방법]
1. 본페로니 보정(Bonferroni correction)
- 가장 간단한 방법으로, 각 비교에 대해 유의수준을 조정하여 1종 오류를 줄이는 방법입니다.
- 예를 들어, 0.05의 유의수준을 사용하고 3개의 비교를 한다면, 각 비교에 대해 0.05/3 = 0.0167의 유의수준을 사용합니다.
- 단점은 보수적이어서 실제로 유의미한 차이가 있는 경우에도 검정 결과가 유의미하지 않을 수 있습니다.

2. 튜키의 HSD (Tukey's Honestly Significant Difference)
- 집단 간 평균 차이를 비교하는 방법으로, 모든 집단 간의 평균 차이를 동시에 비교합니다.
- 이 방법은 집단 간의 분산이 동일하다는 가정 하에 사용됩니다.

유의수준이란?
- 통계적 가설 검정에서 귀무가설을 기각할 기준이 되는 확률값을 의미합니다.
- 즉, 유의수준이 0.05 라면, 귀무가설이 참일 때, 5%의 확률로 잘못된 결론을 내릴 수 있는 기준을 의미합니다.
- 일반적으로 0.05, 0.01, 0.001 등의 값을 사용합니다.

'''

import numpy as np
from scipy.stats import ttest_ind

np.random.seed(42)

def run_ttest_simulation(n, iterations):
    type1_error_count = 0

    for _ in range(iterations):
        # 실제로는 모두 평균 100인 동일한 그룹
        group_a = np.random.normal(loc=100, scale=10, size=n)
        group_b = np.random.normal(loc=100, scale=10, size=n)
        group_c = np.random.normal(loc=100, scale=10, size=n)

        # 3쌍 비교: A-B, A-C, B-C
        p1 = ttest_ind(group_a, group_b).pvalue
        p2 = ttest_ind(group_a, group_c).pvalue
        p3 = ttest_ind(group_b, group_c).pvalue

        # 유의수준 0.05보다 작은 p-value가 하나라도 있으면 → 1종 오류 발생
        if p1 < 0.05 or p2 < 0.05 or p3 < 0.05:
            type1_error_count += 1

    return type1_error_count / iterations

# 1000 번 실행해보자
error_rate = run_ttest_simulation(30, 1000)
print(f"1종 오류가 발생한 비율: {error_rate:.3f}")


from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd

# 가짜 데이터 준비
group = np.array(['A']*30 + ['B']*30 + ['C']*30)
value = np.random.normal(loc=100, scale=10, size=90)
df = pd.DataFrame({'group': group, 'value': value})

# Tukey HSD 수행
tukey = pairwise_tukeyhsd(endog=df['value'], groups=df['group'], alpha=0.05)
print(tukey)



from statsmodels.stats.multicomp import MultiComparison
from scipy import stats

import numpy as np
import pandas as pd

odors = ['Lavender', 'Rosemary', 'Peppermint']

# 각 향기에서 손님이 머물다 간 시간(분) 데이터
# 향기가 머무는 시간에 영향을 미치는지 확인하자.
minutes_lavender = [10, 12, 11, 9, 8, 12, 11, 10, 10]
minutes_rose = [14, 15, 13, 16, 14, 15, 14, 13, 14, 16]
minutes_mint = [18, 17, 18, 16, 17, 19, 18, 17]
anova_data = pd.DataFrame({
    'odor': np.array(["Lavender"] * 9 + ["Rosemary"] * 10 + ["Peppermint"] * 8),
    'minutes': minutes_lavender + minutes_rose + minutes_mint
})

import seaborn as sns
import matplotlib.pyplot as plt

# 시각화
sns.boxplot(x='odor', y='minutes', data=anova_data, palette='Set2')
plt.title('향기별 머무는 시간')
plt.ylabel('머무는 시간(분)')
plt.xlabel('')
plt.show()

from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(
    endog=anova_data['minutes'],   # 종속변수
    groups=anova_data['odor'],     # 그룹
    alpha=0.05                     # 유의수준
)
print(tukey)


from scipy.stats import levene

# 그룹별로 나눠서 잔차 or 원값 비교
minutes_lavender = np.array([10, 12, 11, 9, 8, 12, 11, 10, 10])
minutes_rose = np.array([14, 15, 13, 16, 14, 15, 14, 13, 14, 16])
minutes_mint = np.array([18, 17, 18, 16, 17, 19, 18, 17])

# Levene Test 수행
stat, p = levene(minutes_lavender, minutes_mint, minutes_rose)

print(f"Levene 검정 통계량: {stat:.4f}")
print(f"p-value: {p:.4f}")

if p > 0.05:
    print("✅ 등분산성 만족 → Tukey HSD 사용 가능")
else:
    print("❌ 등분산성 불만족 → 다른 방법 고려 필요")