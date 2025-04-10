plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']


# 회귀분석 기본코드
from statsmodels.formula.api import ols
import statsmodels.api as sm

model = ols("Petal_Length ~ Petal_Width", data=iris).fit()
print(model.summary())


model = ols("Petal_Length ~ Petal_Width", data=iris).fit()
sm.stats.anova_lm(model)
#             df sum_sq mean_sq F PR(>F)
# Petal_Width 1.0 430.480647 430.480647 1882.452368 4.675004e-86
# Residual 148.0 33.844753 0.228681 NaN NaN
# 유의수준 5% 하에서 F value (1882.5)와 대응하는 p-value 을 고려할 때, 너무 작으므
# 로, 귀무가설을 기각한다.



# 다중회귀분석 실행
model2 = ols("Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width",
data=iris).fit()
print(model2.summary())


# 모델을 추가하거나 빼고싶을 때
# 귀무가설: Reduced Model (변수가 적은 모델) 이 알맞음.
# 대립가설: Full Model 이 알맞음.
# Full model: Petal.Width + Sepal.Length + Sepal.Width
# Reduced model: Petal.Width
# F-검정을 진행

model1 = ols('Petal_Length ~ Petal_Width', data=iris).fit() #mod1
model2 = ols('Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width',
data=iris).fit() #mod2
table = sm.stats.anova_lm(model1, model2) #anova
print(table)
# Full 모델이 두번째로 들어가야 함에 주의
# pval 이 0.05 미만이므로 귀무가설 기각. (Full Model 선택)



# 1변수 그래프: Histogram, Box plot
# 주요 체크 사항 - 각 변수들 중 불균형한 분포가 없는지 확인

# 2변수 그래프: Correlation plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
cols = ["Petal_Length", "Sepal_Length", "Sepal_Width", "Petal_Width"]
corr_mat = iris[cols].corr().round(2)
sns.heatmap(corr_mat, annot=True, cmap=plt.cm.Reds);
plt.show()


# 잔차 그래프와 검정
# 1. 정규성
# Anderson-Darling Test or Shapiro-Wilk Test
# 2. 등분산성
# F test(변수가 2개), Levene, Bartlett

import scipy.stats as stats

# 잔차 뽑아오기
residuals = model2.resid
# 잔차의 합은 0 -> 이유는?
# 선형 회귀 모델은 다음과 같은 오차 제곱합(Residual Sum of Squares) 을 최소화하는 방식으로 구현됨
sum(residuals)

fitted_values = model2.fittedvalues

plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)
plt.xlabel("residuals")
plt.ylabel("fitted value");

'''
Scatter plot
1. 선형성(linearity) 가정 확인
잔차들이 예측값(fitted values)에 대해 특정한 패턴 없이 퍼져 있어야 해요.

→ 즉, 랜덤하게 흩어진 모양이어야 해요 (구름처럼 퍼진 형태).
만약 곡선 형태나 U자 형태로 보인다면:
선형 회귀는 적절하지 않다는 의미예요.
비선형 관계를 의심해야 합니다.

2. 등분산성(Homoscedasticity) 가정 확인
잔차의 크기가 예측값이 커짐에 따라 점점 커지거나 작아지는 패턴이 없어야 해요.

만약 잔차가 깔때기 모양(작다가 커지는 등)을 보인다면:

분산이 일정하지 않다는 뜻 (이걸 이분산성이라고 해요)

이러면 회귀 결과의 신뢰성이 떨어져요.

'''



plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt);
plt.show()




# Breusch–Pagan / Cook–Weisberg 검정
# 아이디어: 잔차가 등분산을 갖는단 의미는 독립변수에 의하여 설명이 안된다는 뜻

from statsmodels.stats.diagnostic import het_breuschpagan

model = ols('Petal_Length ~ Petal_Width + Sepal_Length + Sepal_Width', data=iris).fit()
bptest = het_breuschpagan(model.resid, model.model.exog)

# bptest: (LM stat, p-value, f-stat, f p-value)
print('BP-test statistics: ', bptest[0])
print("pval: ", bptest[1])
# BP-test statistics:  6.039114919618998
# pval:  0.10972262962330656
# H0: 모든계수가 0이다. (즉, 잔차가 독립변수들과 무관하다)
# HA: 0이 아닌 계수가 존재한다. (잔차가 어떤 독립변수와 관련 있다.)
# 즉, 귀무가설이 기각되면 등분산성 가정이 틀린다.


# 잔차 독립성: 특정패턴을 띄지않는지, 분산이 변하진않는지 체크
# Durbin-Watson test 실시
#  귀무가설: 잔차들간의 상관성이 존재하지 않는다.
#  대립가설: 잔차들간의 자기 상관성이 존재한다
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(model2.resid)
print(dw_stat)
# dw_stat = 2.0 이면 잔차가 독립적이다.
# dw_stat < 2.0 이면 잔차가 양의 자기상관관계가 있다.
# dw_stat > 2.0 이면 잔차가 음의 자기상관관계가 있다.
# 1.5 ~ 2.5 이면 잔차가 독립적이다.




# 연습문제

import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())

np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)

# 1. train_index를 사용해서 펭귄 데이터에서 인덱스에 대응하는 표본들을 뽑아서 
# train_data를 만드세요. (단, 결측치가 있는 경우 제거)
train_data = penguins.iloc[train_index, :]
train_data.isna().sum()
train_data = train_data.dropna()



# 2. train_data 의 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)를
#  사용하여 산점도를 그려보세요.
sns.scatterplot(x=train_data['bill_depth_mm'], y=train_data['bill_length_mm'])

# 3. 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)의 상관계수를 구하고, 
# 두 변수 사이에 유의미한 상관성이 존재하는지 검정해보세요.
model = ols('bill_length_mm ~ bill_depth_mm', data=train_data).fit
# 귀무가설: 두 변수 사이에 상관성이 존재하지 않는다.
# 대립가설: 두 변수 사이에 상관성이 존재한다. 
from scipy.stats import pearsonr
corr_coef, p_value = pearsonr(train_data['bill_length_mm'],
                              train_data['bill_depth_mm'])
print(f"상관계수: {corr_coef:.3f}")
print(p_value)
# pvalue 가 낮으므로 O


# 4. 펭귄 부리길이 (bill_length_mm)를 부리 깊이 (bill_depth_mm)를 사용하여 설명하는 
# 회귀 모델을 적합시킨 후 2번의 산점도에 회귀 직선을 나타내 보세요. (모델 1)
sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                edgecolor='w', s=50)
x_values = train_data['bill_depth_mm']
y_values = 55.4110 - 0.7062 * x_values
plt.plot(x_values, y_values, 
         color='red', label='Regression Line')
plt.grid(True)
plt.legend()
plt.show()


# 5. 적합된 회귀 모델이 통계적으로 유의한지 판단해보세요.
print(model.summary())
# 유의수준 5%하에서 F 검정 통계량 값 12.93에 대응하는 p‐value값 0.000409에 비추어 보았을 때, 
# 회귀 모델은 통계적으로 유의한 것으로 판단한다.


# 잔차 등분산성 검정 (bptest)
bptest = het_breuschpagan(model.resid, model.model.exog)
bptest  # (LM stat, p-value, f-stat, f p-value)
pval = bptest[1]
pval < 0.05     
# H0: 모든계수가 0이다. (즉, 잔차가 독립변수들과 무관하다)
# HA: 0이 아닌 계수가 존재한다. (잔차가 어떤 독립변수와 관련 있다.)


# 6. 𝑅 값을 구한 후 의미를 해석해 보세요
# 0.062
# 너무작다. 부리 깊이로 부리 길이를 설명할 수 있는 설명력이 6.2% 밖에 되지않는다.
# 추가변수를 사용해서 모델의 설명력을 높이는 것을 고려하자.


# 7. 적합된 회귀 모델의 계수를 해석해 보세요.
# 절편과 기울기가 모두 유의하게 나오지만, 각 변수의 뜻을 고려하면, 절편의 해석은 무의미하다. 
#   (부리 깊이 0인 경우, 부리 길이 56 mm)

# 기울기‐0.7062 값의 의미는, 팔머 펭귄의 경우 부리 깊이가 1mm 증가 할 때, 
# 부리 길이는 평균적으로 0.7062 mm 만큼 감소하는 경향을 보인다고 해석할 수 있다.



# 8. 
# 1번에서 적합한 회귀 모델에 새로운 변수 (종 - species) 변수를 추가하려고 합니다. 성별 변수 정
# 보를 사용하여 점 색깔을 다르게 시각화 한 후 적합된 모델의 회귀 직선을 시각화 해보세요. (모델
# 2)

model2 = ols('bill_length_mm ~ bill_depth_mm + species', data=train_data).fit()
print(model2.summary())

train_data['fitted'] = model2.fittedvalues
# 산점도 (실제 데이터)
sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)

# 그룹별(facet별)로 fitted 선 그리기
for species, df in train_data.groupby('species'):
    df_sorted = df.sort_values('bill_depth_mm')  # X축 기준 정렬
    sns.lineplot(data=df_sorted,
                 x='bill_depth_mm', y='fitted', color="red")
plt.title("Regression Lines(fitted)")
plt.legend()
plt.show()




# 9. 종 변수가 새로 추가된 모델 2가 모델 1 보다 더 좋은 모델이라는 근거를 제시하세요.
sm.stats.anova_lm(model, model2)
# p_value 가 작다. (귀무가설 기각, Full model 선택)


# 10. 모델 2의 계수에 대한 검정과 그 의미를 해석해 보세요.
print(model2.summary())


# 11. 모델 2 에 잔차 그래프를 그리고, 회귀모델 가정을 만족하는지 검증을 수행해주세요.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(model2.fittedvalues, model2.resid)
# 왼쪽과 오른쪽에 클러스터처럼 모여 있음

plt.subplot(1, 2, 2)
stats.probplot(model2.resid, plot=plt)
plt.show()

bptest = het_breuschpagan(model2.resid, model2.model.exog)
if bptest[1] < 0.05:
    print("bptest 에서 pvalue 값이 0.05보다 작습니다. 등분산성을 만족하지 않습니다.")
else:
    print("pvalue 값이 0.05보다 큽니다. 등분산성을 만족합니다.")

from scipy.stats import anderson
# 귀무가설: 데이터는 정규분포를 따른다. / 대립가설: 정규분포를 따르지않는다.
anderson(model2.resid, dist='norm')
# 통계량이 임계값보다 작으면 만족. (기각 못 함)
# 정규분포를 따른다고 할 수 있다. (기각 못 함)






'''

연습문제 모음

'''
import numpy as np
from scipy.stats import pearsonr
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 13])
# 피어슨 상관계수와 p-value 계산
corr_coef, p_value = pearsonr(x, y)
print(f"상관계수: {corr_coef:.3f}")
print(f"p-value: {p_value:.3f}")

# p-value 가 0.05보다 작으므로 귀무가설 기각. 상관관계가 있다.
# 0.974 만큼 상관관계가 있다. (강한 상관관계)



# 2. 
x = np.array([1, 2, 3, 4, 10, 11, 12])
y = np.array([2, 4, 6, 8, 100, 200, -100])

# 이상치가 포함된 경우
corr_coef, p_val = pearsonr(x, y)
print(f"상관계수 (이상치 포함): {corr_coef:.3f}")
print(f"p-value (이상치 포함): {p_val:.3f}")
# 기각. 상관관계가 없다.

# 이상치를 제외한 경우
x_no_outliers = np.array([1, 2, 3, 4])
y_no_outliers = np.array([2, 4, 6, 8])
corr_coef, p_val = pearsonr(x_no_outliers, y_no_outliers)
print(f"상관계수 (이상치 제외): {corr_coef:.3f}")
print(f"p-value (이상치 제외): {p_val:.3f}")
# 강한 양의 상관관계가 있습니다.

# 이상치가 상관계수에 미치는 영향
print("이상치가 상관계수에 큰 영향을 미쳐 왜곡될 수 있습니다.")


# 3.
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 6, 9, 12, 15])

# 단순 선형회귀 계수 B0, B1을 직접 계산하시오.
# 회귀직선 방정식을 구하시오.
B1 = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
B0 = np.mean(y) - B1 * np.mean(x)

print(f"B0: {B0:.2f}")
print(f"B1: {B1:.2f}")
print(f"회귀직선 방정식: y = {B0:.2f} + {B1:.2f}x")




# 4.
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])
# 상관계수를 이용하여 B1_hat을 계산하시오.
# B1_hat = r * (sy / sx)
B1_hat = pearsonr(x, y)[0] * (np.std(y, ddof=1) / np.std(x, ddof=1))
print(f"B1_hat: {B1_hat:.2f}")
print("직접 비교한 결과와 같다.")


# 5. 
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.datasets import fetch_california_housing
cal = fetch_california_housing(as_frame=True)
df = cal.frame
# MedHouseVal을 종속변수로 하고, AveRooms, AveOccup을 독립변수로 설정한 선형회귀모형을 적합하시오.
# 모델의 회귀식을 구하시오.
model = ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
print(model.summary())
# 회귀식: MedHouseVal = -0.0708 * AveRooms + -0.0026 * AveOccup + 1.6919

# p-value ㅔ들은 AveOccup 만 0.001 이고, 나머지는 다 0.000 이므로 유의미하다.


# 6.
model = ols('MedHouseVal ~ AveRooms + AveOccup', data=df).fit()
# t-value 가 가장 큰 변수는 무엇인가?
# 해당 변수의 p-value 는 얼마인가?
# tsatistic 가 가장 큰 변수는 AveRooms 이고, p-value 는 0.000 이다.


# 7.
df['IncomeLevel'] = pd.qcut(df['MedInc'], q=3, labels=['Low', 'Mid', 'High'])
model = ols('MedHouseVal ~ AveRooms + AveOccup + C(IncomeLevel)', data=df).fit()
# 범주형 변수 IncomeLevel에서 가장 유의미한 더미 변수는 무엇인가?
print(model.summary())  # C(IncomeLevel)[T.High] , 1.6558
# 해당 변수의 회귀 계수를 구하세요.



# 8.
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(model.resid)
print(dw_stat)
# dw_stat = 2.0 이면 잔차가 독립적이다.
# dw_stat < 2.0 이면 잔차가 양의 자기상관관계가 있다.
# dw_stat > 2.0 이면 잔차가 음의 자기상관관계가 있다.

# dw_stat 이 0.6인경우
# 잔차가 양의 자기상관관계가 있다.
# 자기 상관이 있을 경우, 회귀모형의 신뢰성이 떨어진다.


# 9.
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(model.resid, model.model.exog)
print(bp_test)  # (LM stat, p-value, f-stat, f p-value)
bp_test[1] < 0.05
# 등분산성 위배
# 위배되면, 회귀모형의 신뢰성이 떨어진다.



# 10.
from statsmodels.stats.outliers_influence \
import variance_inflation_factor
X = df[['AveRooms', 'AveOccup', 'HouseAge']]
vif_df = pd.DataFrame()
vif_df['Variable'] = X.columns
# 계산

# .vif가 