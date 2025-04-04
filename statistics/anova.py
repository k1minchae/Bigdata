import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, f
import seaborn as sns
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 그룹이 1개, 2개인 경우 T검정으로 체크
# 3개 이상의 집단에서 평균의 차이 검증은? ANOVA를 이용한다.

# 1. 일원분산분석 (One-way ANOVA)
# 일원분산분석의 아이디어만 확인하고 넘어가자.

# 모델 가정
# i번째 집단의 j번째 관찰값 = i번째 집단의 평균 + 오차항(잡음)
# i번째 집단의 평균 = 전체 평균 + 집단효과

# 등분산성
# 모든 그룹에서의 오차항의 분산이 동일해야 한다.
# 집단 간 분산 = 집단 내 분산

# 독립성
# 각 집단의 관찰값은 서로 독립적이어야 한다.
# 즉, 한 집단의 관찰값이 다른 집단의 관찰값에 영향을 미치지 않아야 한다.

# 분산분석에서는 
# 1. 잔차의 정규성 
# 2. 잔차의 등분산성
# 3. 데이터의 독립성을 만족해야 한다.



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

anova_data.head()

# F = 그룹외 변동성 / 그룹내 변동성
# 귀무가설: 그룹별 평균이 동일하다. (F통계량이 작다.)
# 대립가설: 그룹별 평균이 동일하지 않다.    (F통계량이 크다.)
# F통계량이 클수록 귀무가설을 기각할 가능성이 높다.
# 결정계수

# 각 그룹의 평균
anova_data.groupby('odor').mean()  
# Lavender: 10.3, Rosemary: 14.4, Peppermint: 17.5

# 시각화
sns.boxplot(x='odor', y='minutes', data=anova_data, palette='Set2')
plt.title('향기별 머무는 시간')
plt.ylabel('머무는 시간(분)')
plt.xlabel('')
plt.show()


# 평균을 맞춰서 다시 시각화해보자.
sub_data = anova_data.copy()
sub_data.loc[sub_data['odor'] == 'Lavender', 'minutes'] += 7
sub_data.loc[sub_data['odor'] == 'Rosemary', 'minutes'] += 3
sns.boxplot(x='odor', y='minutes', data=sub_data, palette='Set2')
plt.title('향기별 머무는 시간 (평균 맞춤)')
plt.ylabel('머무는 시간(분)') 
plt.xlabel('')
plt.show()

# sst = ssg + sse 가 같은지 확인해보기
# sst: 총 변동성, ssg: 집단 간 변동성, sse: 집단 내 변동성
group_means = anova_data.groupby('odor')['minutes'].mean()
group_vars = anova_data.groupby('odor')['minutes'].var(ddof=1)
group_sizes = anova_data['odor'].value_counts()
total_mean = anova_data['minutes'].mean()

sst = anova_data['minutes'].var(ddof=1) * (len(anova_data) - 1)
ssg = ((group_means - total_mean) ** 2 * group_sizes).sum()
sse = (group_vars * (group_sizes - 1)).sum()
ssg + sse, sst
# 잔차가 정규성을 만족하는지 확인
# 잔차: 실제값 - 예측값



# 분산분석을 통해 집단 간 평균 차이를 검정하는 방법은 다음과 같다.
# 1. 각 집단의 평균을 계산한다.
# 2. 전체 평균을 계산한다.
# 3. 각 집단의 평균과 전체 평균의 차이를 계산한다.
# 4. 각 집단의 분산을 계산한다.
# 5. 각 집단의 분산과 전체 분산의 차이를 계산한다.
# 6. F 통계량을 계산한다.
# 7. F 분포를 이용하여 p-value를 계산한다.
# 8. 유의수준과 p-value를 비교하여 귀무가설을 기각할지 결정한다.


lavender = anova_data[anova_data['odor'] == 'Lavender']['minutes']
rosemary = anova_data[anova_data['odor'] == 'Rosemary']['minutes']
peppermint = anova_data[anova_data['odor'] == 'Peppermint']['minutes']

# 일원 분산분석(One-way ANOVA) 수행
f_statistic, p_value = f_oneway(lavender, rosemary, peppermint)
print(f"F 통계량: {f_statistic}, p-value: {p_value}")
p_value < 0.05  # True
# p-value가 0.05보다 작으므로 귀무가설을 기각한다.
# 즉, 향기별 머무는 시간의 평균이 서로 다르다.




# 가정 체크와 여타 분석까지 해야하는 경우는, 앞에서 배운 f_oneway() 함수를 사용하기 보다는
# statmodels 패키지를 사용하는 것이 좋습니다.
# statsmodels 패키지는 회귀분석을 위한 패키지로, ANOVA 분석도 지원합니다.
# statsmodels 패키지를 사용하면 ANOVA 분석을 수행할 수 있습니다.
import statsmodels.api as sm
from statsmodels.formula.api import ols

res_vec = model = ols('minutes ~ C(odor)', data=anova_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
