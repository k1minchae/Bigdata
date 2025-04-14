# 집가격 데이터 불러오기
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import anderson


plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = pd.read_csv('../practice-data/house-data/train.csv')
df.head()
df.info()

'''
조별로 y타겟을 가장 잘 설명하는 선형 회귀모델 구축하기
변수별 관계를 파악한 후 선형 회귀모델 구축

'''

'''
SalePrice        1.000000
OverallQual      0.790982
GrLivArea        0.708624
GarageCars       0.640409
GarageArea       0.623431
TotalBsmtSF      0.613581
1stFlrSF         0.605852
FullBath         0.560664
TotRmsAbvGrd     0.533723
YearBuilt        0.522897
YearRemodAdd     0.507101
GarageYrBlt      0.486362
MasVnrArea       0.477493
Fireplaces       0.466929
BsmtFinSF1       0.386420
LotFrontage      0.351799
WoodDeckSF       0.324413
2ndFlrSF         0.319334
OpenPorchSF      0.315856
HalfBath         0.284108
LotArea          0.263843
BsmtFullBath     0.227122
BsmtUnfSF        0.214479
BedroomAbvGr     0.168213
ScreenPorch      0.111447

수치형 변수. 상관계수

'''
# 결측치 제거
df.isna().sum().sort_values(ascending=False).head(20)
cols_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'LotFrontage']
df = df.drop(columns=cols_to_drop)

# salesprice 데이터 분포 확인
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['SalePrice'], bins=50)
plt.title("집 가격 히스토그램")

df = df.loc[df['SalePrice'] <= 450_000]
plt.subplot(1, 2, 2)
plt.hist(df['SalePrice'], bins=50)
plt.title("집 가격 히스토그램-이상치 제거")



df['Street'].value_counts()
df['MSSubClass'].unique()
df['1stFlrSF'].unique()
df['GarageType'].value_counts()

# TotRmsAbvGrd(), GarageArea 변수 제거
model1 = ols('SalePrice ~ OverallQual + GrLivArea + GarageCars + TotalBsmtSF + Q("1stFlrSF") + FullBath + YearBuilt + YearRemodAdd', data=df).fit()
model2 = ols('SalePrice ~ OverallQual + GrLivArea + GarageCars + GarageArea + TotalBsmtSF + Q("1stFlrSF") + FullBath + TotRmsAbvGrd + YearBuilt + YearRemodAdd', data=df).fit()

print(model1.summary())
print(model2.summary())

anova_result = sm.stats.anova_lm(model1, model2)
print(anova_result)

# TotalBsmtSF(지하실 면적), GarageArea(차고면적) 는 제거해도 괜찮다.
# 잔차 분석
plt.figure(figsize=(7, 5))

# scatter (등분산성)
plt.scatter(model1.fittedvalues, model1.resid)
plt.axhline(0, color='red', linestyle='--')
plt.title("잔차 분석")
plt.xlabel("예측값")
plt.ylabel("잔차")

# 정규성
stats.probplot(model1.resid, plot=plt)
plt.show()




category_model = ols('SalePrice ~ C(Condition1)', data=df).fit()
print(category_model.summary())


full = ols('SalePrice~OverallQual+GrLivArea+GarageCars+GarageArea+TotalBsmtSF+Q("1stFlrSF")+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd+C(KitchenQual)+C(BsmtExposure)+C(BsmtQual)+C(Foundation)+C(ExterQual)+C(HouseStyle)+C(LotShape)'
            ,data=df).fit()
print(full.summary())

# 상관관계 0.5 이상 + 범주형
small = ols('SalePrice~OverallQual+GrLivArea+GarageCars+Q("1stFlrSF")+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd+C(KitchenQual)+C(BsmtExposure)+C(BsmtQual)+C(Foundation)+C(ExterQual)+C(HouseStyle)+C(LotShape)'
            ,data=df).fit()
print(small.summary())

# 상관관계 0.3 이상 + 범주형
small2 = ols('SalePrice~OverallQual + MasVnrArea + BsmtFinSF1 + TotalBsmtSF + Fireplaces + OpenPorchSF + WoodDeckSF + GarageArea + GarageYrBlt + Q("2ndFlrSF") + GrLivArea + GarageCars+Q("1stFlrSF")+FullBath+TotRmsAbvGrd+YearBuilt+YearRemodAdd+C(KitchenQual)+C(BsmtExposure)+C(BsmtQual)+C(Foundation)+C(ExterQual)+C(HouseStyle)+C(LotShape)'
            ,data=df).fit()
print(small2.summary())

anova_result = sm.stats.anova_lm(small, small2)
print(anova_result)


########################



residuals = model1.resid
# 평균과 표준편차
mean = residuals.mean()
std = residuals.std()

# ±2표준편차 범위를 벗어나는 값 식별 (이상치)
outlier_mask = (residuals < mean - 2 * std) | (residuals > mean + 2 * std)

# 이상치 제거한 데이터프레임 생성
cleaned_data = df.loc[~outlier_mask].copy()
clean_model = ols('SalePrice ~ OverallQual + GrLivArea + GarageCars + TotalBsmtSF + Q("1stFlrSF") + FullBath + YearBuilt + YearRemodAdd', data=cleaned_data).fit()

stats.probplot(clean_model.resid, plot=plt)
plt.show()
print(clean_model.summary())


'''
Step wise 방법 : 단계별로 변수를 추가해나가면서 최적의 모델을 찾는 방법.
보통은 추가할거랑, 제거할거를 p-value 값을 0.15를 기준으로 선택함

- 자동으로 베스트 모델을 찾아줌.
- R square 값이 자동으로 높게 측정되는 경향성을 띈다.
- 신뢰구간이 비정상적으로 높게나오는 구간 선택됨
- 다중공선성이 문제가 되는 데이터의 경우 잘 작동하지 않는다.

AIC, BIC 
Base stepwise method
'''


