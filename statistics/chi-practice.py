# [연습문제] 카이제곱 검정 이해하기 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import scipy.stats as stats
import seaborn as sns

path = "../practice-data/estimate-data/"


'''
문제 1

1. 귀무가설과 대립가설을 설정하시오.
2. 가설을 확인하기 위해 Pregnancy_status와 Outcome의 관계를 시각화하시오.
3. Pregnancy_status와 Outcome에 대한 교차표를 만들고, 각 셀의 기대빈도를 계산하시오.
4. 카이제곱 독립성 검정을 수행하시오.
    - 검정통계량, 자유도, 유의확률(p-value)을 구하시오.
5. 유의수준 5%에서 귀무가설의 기각 여부를 판단하고 해석하시오.

'''
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
dat = pd.read_csv(url, header=None, names=col_names)
# 임신 유무 파생변수 생성
dat['Pregnancy_status'] = (dat['Pregnancies'] > 0).astype(int)
dat.head()

# 데이터 전처리 (필요한 컬럼만 남김)
dat1 = dat.loc[:, ['Pregnancy_status', 'Outcome']]

# 1. 가설 설정
# H0: 임신유무와 당뇨병 여부는 독립이다.
# H1: 임신유무와 당뇨병 여부는 독립이 아니다.


# 3. 교차표 만들기
cross_table = pd.crosstab(dat1['Pregnancy_status'], dat1['Outcome'])
print(cross_table)

# 2. 시각화 (누적 막대그래프)
# 교차표를 비율로 변환
cross_table_ratio = cross_table.div(cross_table.sum(axis=1), axis=0)

cross_table_ratio.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.ylabel('Proportion')
plt.title('Pregnancy Status vs Outcome (Stacked Proportion)')
plt.xticks(ticks=[0, 1], labels=['No Pregnancy', 'Pregnancy'], rotation=0)
plt.legend(title='Outcome', loc='upper right', labels=['X', 'O'])
plt.show()


# 3-1. 각 셀의 기대빈도 계산
chi2, p, df, expected = chi2_contingency(cross_table, correction=False)
print(f'Expected Frequencies:\n{expected}')

# 4. 카이제곱 독립성 검정 결과 출력
print(f'검정통계량: {chi2:.2f}')
print(f'자유도: {df}')
print(f'P-value: {p:.2f}')

# 5. 유의수준 5%에서 귀무가설 기각 여부 판단 (기각X, 독립O: 임신/당뇨 무관)
alpha = 0.05
if p < alpha:
    print("귀무가설을 기각합니다. 임신유무와 당뇨병 여부는 독립이 아닙니다.")
else:
    print("귀무가설을 기각하지 않습니다. 임신유무와 당뇨병 여부는 독립입니다.")


'''
문제 2

1. 귀무가설과 대립가설을 설정하시오.
2. 가설을 확인하기 위해 Age_group와 Outcome의 관계를 시각화하시오.
3. Age_group과 Outcome 간의 교차표를 만들고, 기대빈도를 계산하시오.
4. 카이제곱 동질성 검정을 수행하시오.
5. 유의수준 5%에서 귀무가설 기각 여부를 판단하고 해석하시오.

'''
# 1. 가설 설정
# H0: 연령대와 당뇨병 여부는 독립이다.
# H1: 연령대와 당뇨병 여부는 독립이 아니다.

# 2. Age Group 생성
dat.info()
dat['Age_group'] = np.where(dat['Age'] >= 40, 'Over40', 'Under40')


# 3. 교차표 만들기
cross_table2 = pd.crosstab(dat['Age_group'], dat['Outcome'])
print(cross_table2)


# 2. 시각화
cross2_ratio = cross_table2.div(cross_table2.sum(axis=1), axis=0)
cross2_ratio.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.ylabel('Proportion')
plt.title('Age Group vs Outcome (Stacked Proportion)')
plt.xticks(rotation=0)
plt.legend(title='Outcome', loc='upper right', labels=['X', 'O'])
plt.show()



# 각 셀의 기대빈도 계산
chi2, p, df, expected = chi2_contingency(cross_table2, correction=False)
print(f'각 셀의 기대 빈도:\n{expected}')


# 4. 카이제곱 동질성 검정 결과 출력
print(f'검정통계량: {chi2:.2f}')
print(f'자유도: {df}')
print(f'P-value: {p:.2f}')


# 5. 유의수준 5%에서 귀무가설 기각 여부 판단 (기각: 독립아님 => 연령-당뇨 관계성 입증O)
alpha = 0.05
if p < alpha:
    print("귀무가설을 기각합니다. 연령대와 당뇨병 여부는 독립이 아닙니다.")
else:
    print("귀무가설을 기각하지 않습니다. 연령대와 당뇨병 여부는 독립입니다.")




'''
문제 3

1. 귀무가설과 대립가설을 설정하시오.
2. 교차표 데이터를 바탕으로 카이제곱 독립성 검정을 수행하시오.
3. 검정통계량, 자유도, p-value를 구하고 해석하시오.

'''

# 1. 가설 검정
# H0: 운동 빈도와 건강 상태는 독립이다.
# H1: 운동 빈도와 건강 상태는 독립이 아니다.

# 운동 빈도(안함, 주1-2회, 주3회)와 건강 상태(Healthy, Unhealthy) 간의 교차표
cross_table3 = ([[30, 70],
                 [50, 50],
                 [70, 30]]) 

# 시각화
# 교차표를 비율로 변환
cross3_ratio = cross_table3 / np.sum(cross_table3, axis=1, keepdims=True)
cross3_ratio = pd.DataFrame(cross3_ratio, columns=['Healthy', 'Unhealthy'], index=['No Exercise', '1-2 times/week', '3 times/week'])
cross3_ratio.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.ylabel('Proportion')
plt.title('Exercise Frequency vs Health Status')
plt.xticks(rotation=0)
plt.legend(title='Health Status', loc='upper right', labels=['Healthy', 'Unhealthy'])
plt.show()


# 2. 카이제곱 검정 수행
chi2, p, df, expected = chi2_contingency(cross_table3, correction=False)
print(f'검정통계량: {chi2:.2f}')
print(f'자유도: {df}')
print(f'P-value: {p:.2f}')

# 3. 유의수준 5%에서 귀무가설 기각 여부 판단 (기각O, 독립X: 운동-건강 관계성 입증O)
alpha = 0.05
if p < alpha:
    print("귀무가설을 기각합니다. 운동 빈도와 건강 상태는 독립이 아닙니다.=> 관계 O")
else:
    print("귀무가설을 기각하지 않습니다. 운동 빈도와 건강 상태는 독립입니다.")



'''
문제 4

소규모 건강 설문에서 60명의 사람들을 대상으로 식습관과 건강 상태를 조사했습니다. 
설문 결과는 다음과 같습니다.

1. 귀무가설과 대립가설을 설정하시오.
2. 교차표 데이터를 바탕으로 카이제곱 독립성 검정을 수행하시오.
3. 기대빈도를 만족시키기 위해 범주를 병합하고 다시 카이제곱 검정을 수행하시오.
4. 유의수준 5%에서 귀무가설 기각 여부를 판단하고 해석하시오.

'''
# 교차표 데이터 (식습관과 건강 상태)
# 식습관: 자주거름, 불규칙, 하루2끼, 규칙적
# 건강: 건강, 비건강
cross_4 = ([[1, 4],
            [2, 8],
            [4, 6],
            [15, 20]])

# 데이터 병합
# 자주거름, 불규칙, 하루2끼 -> 불규칙식사
# 하루2끼, 규칙적 -> 규칙식사
# 교차표 데이터 (병합 후)
cross_4_merged = ([[7, 18],
                    [15, 20]])


# 시각화 (누적 막대그래프)
cross4_ratio = cross_4_merged / np.sum(cross_4_merged, axis=1, keepdims=True)
cross4_ratio = pd.DataFrame(cross4_ratio, columns=['Healthy', 'Unhealthy'], index=['Irregular Eating', 'Regular Eating'])
cross4_ratio.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.ylabel('Proportion')
plt.title('Eating Habits vs Health Status (Stacked Proportion)')
plt.xticks(rotation=0)
plt.legend(title='Health Status', loc='upper right', labels=['Healthy', 'Unhealthy'])
plt.show()


# 1. 가설 검정
# H0: 식습관과 건강 상태는 독립이다.
# H1: 식습관과 건강 상태는 독립이 아니다.

# 2. 카이제곱 검정 수행
chi2, p, df, expected = chi2_contingency(cross_4_merged, correction=False)
print(f'검정통계량: {chi2:.2f}')
print(f'자유도: {df}')
print(f'P-value: {p:.2f}')

# 3. 유의수준 5%에서 귀무가설 기각 여부 판단 (기각X, 독립O: 식습관-건강 관계 없음)
alpha = 0.05
if p < alpha:
    print("귀무가설을 기각합니다. 식습관과 건강 상태는 독립이 아닙니다.")
else:
    print("귀무가설을 기각하지 않습니다. 식습관과 건강 상태는 독립입니다.")
