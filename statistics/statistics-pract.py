# 통계 전체 연습문제

# 문제 1
import pandas as pd
from scipy import stats
df = pd.read_csv('../practice-data/estimate-data/maching.csv')
print(df.head(3))

# A 회사는 이벤트를 활용한 마케팅이 효과가 있는지 검증하고자 한다.
# event : 이벤트 참여 여부
# age : 나이
# revenue : 수익

# 이벤트 참여 여부에 따른 평균 수익의 차이를 계산하고 해석하시오.
# 독립 표본 t검정
event = df.loc[df['event']== 1, 'revenue']
non_event = df.loc[df['event']== 0, 'revenue']
stats.levene(event, non_event) # 등분산성 검정 => 등분산

stat, pval = stats.ttest_ind(event, non_event, equal_var=True, alternative='two-sided')
# 이벤트 참여 여부에 따른 수익 차이가 있다.


# 이벤트 참여 여부에 따른 평균 수익의 분포를 시각화하시오.
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(x='event', y='revenue', data=df, palette='Set2')
plt.title('Revenue Distribution by Event Participation')
plt.xlabel('Event Participation (0 = No, 1 = Yes)')
plt.ylabel('Revenue')
plt.show()




# 이벤트 참여 여부에 따른 평균 수익의 차이를 통계적으로 검증하시오.
# 추가 분석을 자유롭게 수행한 후 종합적인 결론을 도출하시오.
sns.lmplot(data=df, x='age', y='revenue', hue='event', palette='Set2')
plt.show()
# 나이가 높을수록 연봉이높다.
# 나이가 높을수록 이벤트를 참여할 확률이 높다.

# 나이로 인한 효과때문에 이벤트 참여 여부에 따른 수익 차이가 발생할 수 있다.
# 따라서 나이를 통제한 후 이벤트 참여 여부에 따른 평균 수익의 차이를 검증해야 한다.





# 문제 2
df = pd.read_csv('../practice-data/estimate-data/problem5_27.csv')
df
# 귀무가설과 대립가설을 설정하시오.
# 검정 통계량과 유의 확률을 구하고, 귀무가설 기각 여부를 판단하시오.
# 대응표본 t검정 => 




# 문제 3
dat = pd.read_csv('../practice-data/estimate-data/problem4_30.csv')

# 개인의 헤드셋 보유 정보를 포함하는 데이터입니다.

# index: 인덱스
# headset: 헤드셋 종류
# age: 연령대

# 헤드셋에 대한 연령대별 선호도 차이가 있는지를 유의 수준 5%로 검정하시오. (단, 반올림하여 소수점 셋째자리까지 표시하시오)
# 연구 가설(H1)과 귀무 가설(H0)을 설정하시오.
# 유의 확률을 계산하고 가설의 채택 여부를 결정하시오.

# 카이제곱검정: 범주형 변수 간의 독립성 검정
# 귀무가설(H0): 연령대와 헤드셋 선호도는 독립적이다.
# 대립가설(H1): 연령대와 헤드셋 선호도는 독립적이지 않다.
# 👉 두 변수 모두 범주형이기 때문에,
# → ✔️ 카이제곱 검정이 적절합니다.

from scipy.stats import chi2_contingency
cross_table = pd.crosstab(dat['age'], dat['headset']) # 교차표 생성
result = chi2_contingency(cross_table) # 카이제곱 검정 수행
stat = result.statistic
pval = result.pvalue
# 결과 출력
print(f"F-statistic: {stat:.3f}, p-value: {pval:.3f}")

# 가설 채택 여부 판단
if pval < 0.05:
    print("귀무가설을 기각합니다. 연령대별 헤드셋 선호도에 차이가 있습니다.")
else:
    print("귀무가설을 채택합니다. 연령대별 헤드셋 선호도에 차이가 없습니다.")




# 문제 4
# name : 차종(A, B, C, D)
# ratio : 5회 실험 시 범퍼 파손 정도

df = pd.read_csv('../practice-data/problem5_29.csv')
# 각 차종 별 범퍼 파손의 정도에 차이가 유의한지 검정하시오(모분산 동일, 정규성 가정 하에).
# 귀무가설과 대립가설을 설정하시오.
# H0: 차종에 따른 범퍼 파손 정도에 차이가 없다.
# H1: 차종에 따른 범퍼 파손 정도에 차이가 있다.

stats.f_oneway(df[df['name'] == 'A']['ratio'],
          df[df['name'] == 'B']['ratio'],
          df[df['name'] == 'C']['ratio'],
          df[df['name'] == 'D']['ratio'])
# 5% 유의수준에서 검정하시오.
# 기각. 차종에 따른 범퍼 파손 정도에 차이가 있다.

# 사후분석까지 수행하시오.


# 사후분석 수행
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey_result = pairwise_tukeyhsd(df['ratio'], df['name'], alpha=0.05)

# 결과 출력
print(tukey_result)
# A-D빼고는 다 차이가 있다.



# 문제 5: 참고용?
# 한 의류 제조사는 두 개의 생산라인(A, B)의 옷감 두께의 일관성(분산)을 비교하고자 한다. 
# 각각의 생산라인에서 무작위로 옷감 샘플을 채취한 결과 다음과 같은 통계 요약이 도출되었다.

# A라인: 표본 크기 = 20, 표본 분산 = 2.5
# B라인: 표본 크기 = 18, 표본 분산 = 1.2
# 생산관리팀은 두 라인의 생산 품질이 동일한 수준의 변동성(분산)을 가지고 있는지 검정하고자 한다. 유의수준 5%에서 이표본 분산 검정을 실시하시오.

# 귀무가설과 대립가설을 설정하시오.
# 적절한 검정 통계량과 자유도를 구하시오.
# 겸정 결과를 해석하시오.
# H0: 두 생산라인의 분산이 같다.
# H1: 두 생산라인의 분산이 다르다.

# F-검정 통계량 계산
fstat = 2.5 / 1.2
# 자유도 계산
df1 = 20 - 1
df2 = 18 - 1

# pvalue 구하기
from scipy.stats import f
pval = f.sf(fstat, df1, df2) * 2
print(pval)
# 0.05보다 크므로, 귀무가설을 기각하지 않는다.
# 두 생산라인의 분산은 같다.


'''
카이제곱 적합도 검정
(이산형)
'''

# 어느 고객지원센터는 시간당 평균 1.2건의 민원이 발생한다고 알려져있다.
# 무작위로 60시간 관찰한 결과, 시간당 민원발생 건수에 대한 빈도는 다음과 같다.
# 0건: 18건, 1건: 20건, 2건: 13건, 3건 이상: 9건
# 이 데이터가 평균 1.2인 포아송분포에서 왔다고 할수있는가?
import numpy as np
from scipy.stats import poisson, chisquare
# 관측 분포와 기대 분포의 적합성 분석: chisquare,
# 두 범주형 변수 간의 관련성 분석:	chi2_contingency

x = np.array([18, 20, 13, 9])  # 관측 빈도
n = x.sum()  # 총 관측 횟수

mu = 1.2  # 포아송 분포의 평균

# 각 범주에 대한 기대 확률 계산
p0 = poisson.pmf(0, mu)
p1 = poisson.pmf(1, mu)
p2 = poisson.pmf(2, mu)
p3 = 1 - (p0 + p1 + p2)  # 3건 이상

# 기대 빈도 계산
expected = n * np.array([p0, p1, p2, p3])

# 카이제곱 검정 수행
stat, pval = chisquare(f_obs=x, f_exp=expected)

# 결과 출력
print(f"Chi-squared statistic: {stat:.3f}, p-value: {pval:.3f}")

# 가설 채택 여부 판단
if pval < 0.05:
    print("귀무가설을 기각합니다. 데이터는 평균 1.2인 포아송분포에서 왔다고 할 수 없습니다.")
else:
    print("귀무가설을 채택합니다. 데이터는 평균 1.2인 포아송분포에서 왔다고 할 수 있습니다.")
