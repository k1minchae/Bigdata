import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f, ttest_rel
import seaborn as sns

path = "../practice-data/estimate-data/"
filename = "problem5_27.csv"
data = pd.read_csv(path + filename)

'''
1번 문제
강의 상류와 하류의 생물 다양성 점수에 차이가 있는지 검정하시오. 
(단, 같은 강에서 상류와 하류는 서로 독립적이지 않다.)

1. 귀무가설과 대립가설을 설정하시오.
2. 가설에 대한 검정 통계량과 유의 확률을 구하고, 
    귀무가설 기각 여부를 판단하시오
'''
# H0: 상류와 하류의 생물 다양성 점수가 같다.
# H1: 상류와 하류의 생물 다양성 점수가 다르다.
up = data['up']
down = data['down']

tstat, pval = ttest_rel(up, down, alternative='two-sided')
pval < 0.05  # False
# 대응표본 t-검정을 사용하여 상류와 하류의 생물 다양성 점수 차이를 검정.
# 귀무가설 기각 X
# 즉, 상류와 하류의 생물 다양성 점수는 차이가 없다.






filename = "problem5_32.csv"
data = pd.read_csv(path + filename)
'''
2번 문제
성별에 따른 급여의 평균 차이가 있는지 검정하시오.

1. 귀무가설과 대립가설을 설정하시오.
2. 성별과 급여의 차이를 알아보기 위해 데이터를 시각화하시오.
3. 그룹 간 분산의 차이가 있는지 검정하시오.
4. 가설에 대한 검정 통계량과 유의 확률을 구하고, 
    귀무가설 기각 여부를 판단하시오.
'''
# H0: 남성과 여성의 급여 평균은 같다.
# H1: 남성의 급여 평균이 더 높다.
male = data.loc[data["Gender"] == "Male", 'Salary']
female = data.loc[data["Gender"] == "Female", 'Salary']

# 박스플롯 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, y='Salary', x='Gender', patch_artist=True, palette="Set2")
plt.title('Salary Distribution by Gender')
plt.suptitle('') 
plt.xlabel('Gender')
plt.ylabel('Salary')
plt.show()



fstat = male.var(ddof=1) / female.var(ddof=1)
p_val = f.sf(fstat, dfn=len(male) - 1, dfd=len(female) - 1) * 2
p_val < 0.05  # False
# 분산이 같다.

tstat, pval = ttest_ind(male, female, alternative='greater', equal_var=True)
pval < 0.05  # True
# 귀무가설을 기각한다.
# 즉, 남성이 여성의 급여 평균보다 더 높다.



filename = "heart_disease.csv"
data = pd.read_csv(path + filename)
'''
3번 문제

심장 질환이 있는 그룹과 없는 그룹의 콜레스테롤 수치의 평균에 차이가 있는지 검정하시오.

1. 귀무가설과 대립가설을 설정하시오.
2. 변수 간 관계를 알아보기 위해 데이터를 시각화하시오.
3. 그룹 간 분산의 차이가 있는지 검정하시오.
4. 가설에 대한 검정 통계량과 유의 확률을 구하고, 대립가설 채택 여부를 판단하시오.

'''
# H0: 심장 질환이 있는 그룹과 없는 그룹의 콜레스테롤 수치의 평균은 같다.
# H1: 심장 질환이 있는 그룹이 콜레스테롤 수치 평균이 더 높다.
data.isna().sum()  # 결측치 확인
disease = data.loc[data['target'] == 'yes', 'chol'].dropna()
non_disease = data.loc[data['target'] == 'no', 'chol'].dropna()

# 박스플롯 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x='target', y='chol', data=data, palette="Set2")
plt.title('Cholesterol Levels by Heart Disease Status', fontsize=14)
plt.xlabel('Heart Disease Status', fontsize=12)
plt.ylabel('Cholesterol', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()


# 분산 동질성 검정
fstat = disease.var(ddof=1) / non_disease.var(ddof=1)
p_val = f.sf(fstat, dfn=len(disease) - 1, dfd=len(non_disease) - 1) * 2
p_val < 0.05  # False
# 분산이 다르지 않다.

# t-검정
tstat, pval = ttest_ind(disease, non_disease, equal_var=True, alternative='greater')
pval < 0.05  # False
# 귀무가설을 기각하지 않는다.
# 즉, 심장 질환이 있는 그룹과 없는 그룹의 콜레스테롤 수치의 평균은 차이가 없다.




'''
4번 문제
당뇨병이 있는 사람과 없는 사람 간의 평균 BMI에 차이가 있는지 검정하시오.

귀무가설과 대립가설을 설정하시오.
변수 간 관계를 알아보기 위해 데이터를 시각화하시오.
그룹 간 분산의 차이가 있는지 검정하시오.
가설에 대한 검정 통계량과 유의 확률을 구하고, 대립가설 채택 여부를 판단하시오.

'''
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
             "DiabetesPedigreeFunction", "Age", "Outcome"]
data = pd.read_csv(url, header=None, names=col_names)
data.isna().sum()  # 결측치 확인
data.head()

disease = data.loc[data['Outcome'] == 1, 'BMI']
non_disease = data.loc[data['Outcome'] == 0, 'BMI']

# 귀무가설: 당뇨병이 있는 사람과 없는 사람의 평균 BMI는 같다.
# 대립가설: 당뇨병이 있는 사람이 평균 BMI가 더 높다.

# 박스플롯 시각화
sns.boxplot(x='Outcome', y='BMI', data=data, palette="Set2")
plt.title('BMI by Diabetes Status', fontsize=14)
plt.show()


# 분산 동질성 검정
fstat = disease.var(ddof=1) / non_disease.var(ddof=1)
p_val = f.cdf(fstat, dfn=len(disease) - 1, dfd=len(non_disease) - 1) * 2
p_val < 0.05  # False
# 분산이 다르지 않다.

# t-검정
tstat, pval = ttest_ind(disease, non_disease, equal_var=True, alternative='greater')
pval < 0.05  # True
# 귀무가설을 기각한다.
# 즉, 당뇨병이 있는 사람의 평균 BMI가 더 높다.





