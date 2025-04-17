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


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 수치형 변수와 범주형 변수 분리
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 결측치 처리
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())  # 수치형 결측치 평균으로 처리
df[categorical_features] = df[categorical_features].fillna('Missing')  # 범주형 결측치 'Missing'으로 처리

# 더미로 변환
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# X (특성)과 y (타겟) 분리
y = df_encoded['SalePrice']
X = df_encoded.drop('SalePrice', axis=1)

# 데이터 스케일링 (선형 회귀 모델 성능 안정화)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled.shape
# AIC 계산 함수
def calculate_aic(model, X, y):
    n = len(y)  # 데이터 샘플 수
    k = X.shape[1]  # 모델 파라미터 수 (특성 개수)
    
    # 모델 학습
    model.fit(X, y)
    
    # 예측값과 실제값의 잔차
    y_pred = model.predict(X)
    residual_sum_of_squares = np.sum((y - y_pred) ** 2)
    
    # AIC 계산
    aic = n * np.log(residual_sum_of_squares / n) + 2 * k
    return aic

# 선형 회귀 모델 정의
lr = LinearRegression()

# 순차적 특성 선택 (SFS) - 특성 수 5개부터 50개까지 비교
sfs = SFS(lr, 
          k_features=(5, 235),         # 5개에서 50개 특성까지 선택
          forward=True,               # 특성을 하나씩 추가하면서 선택
          scoring='r2',  
          cv=0,                       # 교차 검증 사용 안 함
          verbose=2)

# 특성 선택 학습
sfs.fit(X_scaled, y)

# 선택된 특성들 (각 특성 조합에 대해 AIC 계산)
aic_values = []  # AIC 값을 저장할 리스트

# 선택된 특성 조합을 하나씩 확인하고 AIC 계산
for i in range(1, len(sfs.k_feature_idx_) + 1):
    selected_features = np.array(X.columns)[list(sfs.k_feature_idx_[:i])]
    X_selected = X_scaled[:, sfs.k_feature_idx_[:i]]  # 선택된 특성만 사용
    aic_value = calculate_aic(lr, X_selected, y)
    aic_values.append((selected_features, aic_value))

# AIC 값이 가장 작은 모델 찾기
best_model = min(aic_values, key=lambda x: x[1])

print(f"선택된 특성들: {best_model[0]}")
best_model[0].size
print(f"최적 모델의 AIC: {best_model[1]}")





# 여기부터

############## Adj R2
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from sklearn.metrics import r2_score


# 데이터 불러오기
df = pd.read_csv('../data/houseprice/train.csv')

# 타겟과 피처 분리
target = 'SalePrice'
y = df[target]
X = df.drop(columns=[target])

# 수치형 변수만 선택
# X = X.select_dtypes(include=[np.number])

# 수치형과 범주형 구분
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.select_dtypes(include='object').columns

# 결측치 처리
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])


# 범주형 처리
X = pd.get_dummies(X, drop_first=True)




# Adj R2 스코어 함수 정의
def adjusted_r2_score(estimator, X, y):
    y_pred = estimator.predict(X)
    n = X.shape[0]
    p = X.shape[1]
    r2 = r2_score(y, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


# Sequential Feature Selector 설정
lr = LinearRegression()


# Perform SFS: Adj R2_score
sfs = SFS(lr,
          k_features=(1,len(X.columns)),
          forward=True,
          scoring=adjusted_r2_score,
          cv=0,
          verbose = 2)



# 모델 학습
sfs.fit(X, y)

# 선택된 피처 출력
selected_indices_r2 = list(sfs.k_feature_idx_)
names_r2 = np.array(X.columns)[:-1]
print('Selected features:', np.array(names_r2)[selected_indices_r2])

len(selected_indices_r2) # 140


features_r2 = names_r2[list(sfs.k_feature_idx_)]


# ols 
from statsmodels.formula.api import ols

# 리스트를 문자열로 변환해서 formula 작성
target = 'SalePrice'
features_safe_r2 = [f'Q("{col}")' for col in features_r2]
formula_r2 = f"{target} ~ {' + '.join(features_safe_r2)}"

len(features_safe_r2)# # 140

# OLS 모델 적합
df_f = X.copy()
df_f['SalePrice'] = y



model_r2 = ols(formula_r2, data=df_f).fit()
print(model_r2.summary())
# Adj. R0squared: 0.922
# AIC: 3.350e+04



# 캐글돌려보기
# 1. 예측할 test.csv 불러오기
test = pd.read_csv('../data/houseprice/test.csv')
test_ids = test['Id']  # 제출용 ID 저장
test = test.drop(columns='Id')

# 2. train에서 사용했던 수치형/범주형 구분
num_cols_test = test.select_dtypes(include=[np.number]).columns
cat_cols_test = test.select_dtypes(include='object').columns

# 3. 결측치 처리 (train과 동일하게)
test[num_cols_test] = test[num_cols_test].fillna(test[num_cols_test].median())
test[cat_cols_test] = test[cat_cols_test].fillna(test[cat_cols_test].mode().iloc[0])

# 4. 범주형 변수 원-핫 인코딩 (drop_first 포함)
test_dummies = pd.get_dummies(test, drop_first=True)

# 5. train에서 선택된 features_r2 에 맞추어 누락된 열 추가
for col in features_r2:
    if col not in test_dummies.columns:
        test_dummies[col] = 0  # 없는 feature는 0으로 채움

# 6. 순서 맞추기 (model_r2에 사용된 feature 순서와 동일하게)
test_dummies = test_dummies[features_r2]

# 7. DataFrame을 model_r2에 넣기 위한 formula 방식으로 준비
df_test_for_pred = test_dummies.copy()

# 8. 예측
preds = model_r2.predict(df_test_for_pred)
preds = abs(preds)

# 9. 제출용 파일 저장
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': preds})
submission.to_csv('submission_r2.csv', index=False)
print("submission_r2.csv save")
# 점수: 1.74




# AIC
# 1. 수치형 & 범주형 분리
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 2. 결측치 처리
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
df[categorical_features] = df[categorical_features].fillna('-')

# 3. 더미변수 변환
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

y = df_encoded['SalePrice']
X = df_encoded.drop(columns=['SalePrice'])

# 4. 수치형으로 변환
X = X.astype(float)
y = y.astype(float)

# 5. AIC 스코어 함수
def aic_score(estimator, X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return -model.aic

# 6. SFS 실행
lr = LinearRegression()
sfs = SFS(lr,
          k_features='best',
          forward=True,
          scoring=aic_score,
          verbose=2,
          cv=0)

sfs = sfs.fit(X, y)

# 7. 결과 출력
print("선택된 피처 인덱스:", sfs.k_feature_idx_)
print("선택된 피처 이름:", sfs.k_feature_names_)
print("선택된 피쳐 개수: ", len(sfs.k_feature_names_))
print("AIC score:", -sfs.k_score_)

X_const = sm.add_constant(X[list(sfs.k_feature_names_)])
final_model = sm.OLS(y, X_const).fit()

# 중간 과정 확인
selected_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
selected_df = selected_df.sort_index()  # 단계별 보기
selected_df[['feature_names', 'avg_score']]


print(small2.summary()) 
print(final_model.summary()) 





############ 캐글
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
# 정답값 y 분리
y = train["SalePrice"]
train = train.drop(columns=["SalePrice"])

# 병합 후 통합 전처리
train["__is_train__"] = 1
test["__is_train__"] = 0
test["SalePrice"] = np.nan  # 더미 컬럼 추가

df = pd.concat([train, test], axis=0)

# 2. 수치형 / 범주형 분리
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# 3. 결측치 처리
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())
df[categorical_features] = df[categorical_features].fillna("-")

# 4. 더미변수 처리
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# 5. 다시 train/test 분리
train_encoded = df_encoded[df_encoded["__is_train__"] == 1].drop(columns=["__is_train__"])
test_encoded = df_encoded[df_encoded["__is_train__"] == 0].drop(columns=["__is_train__", "SalePrice"])

# X, y 설정
X = train_encoded.drop(columns=["SalePrice"])
y = y.astype(float)
X = X.astype(float)

# 6. AIC 기반 SFS
def aic_score(estimator, X, y):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return -model.aic

lr = LinearRegression()
sfs = SFS(lr,
          k_features='best',
          forward=True,
          scoring=aic_score,
          verbose=2,
          cv=0)

sfs = sfs.fit(X, y)

selected_features = list(sfs.k_feature_names_)
print("선택된 피처 수:", len(selected_features))

# 7. OLS 최종 모델 학습
X_selected = sm.add_constant(X[selected_features])
final_model = sm.OLS(y, X_selected).fit()
print("최종 AIC:", final_model.aic)

# 8. 테스트 데이터 예측
X_test_selected = test_encoded[selected_features].fillna(0)
X_test_selected = sm.add_constant(X_test_selected, has_constant='add')

y_pred = final_model.predict(X_test_selected)

# 9. 제출파일 생성
submit = pd.read_csv("sample_submission.csv")
submit["SalePrice"] = y_pred
submit.to_csv("sfs_aic_submission.csv", index=False)

print("제출 파일 저장 완료: sfs_aic_submission.csv")
# 결과: 0.18




# 라쏘회귀
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. 학습 및 테스트 데이터 로드
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

# 2. 데이터 합치기 (동일한 전처리를 위해)
train["is_train"] = 1
test["is_train"] = 0
test["SalePrice"] = np.nan  # 타겟 임시 생성

df_all = pd.concat([train, test], axis=0)

# 3. 수치형 & 범주형 분리
numeric_features = df_all.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df_all.select_dtypes(include=['object']).columns.tolist()

# 4. 결측치 처리
df_all[numeric_features] = df_all[numeric_features].fillna(df_all[numeric_features].mean())
df_all[categorical_features] = df_all[categorical_features].fillna("-")

# 5. 더미 변수 처리
df_all_encoded = pd.get_dummies(df_all, columns=categorical_features, drop_first=True)

# 6. 학습 / 테스트 분리
train_encoded = df_all_encoded[df_all_encoded["is_train"] == 1].drop(columns=["is_train"])
test_encoded = df_all_encoded[df_all_encoded["is_train"] == 0].drop(columns=["is_train", "SalePrice"])

# 7. 학습용 X, y
X_train = train_encoded.drop(columns=["SalePrice"])
y_train = train_encoded["SalePrice"]

# 8. 라쏘 회귀 학습
lasso_pipeline = make_pipeline(
    StandardScaler(),
    LassoCV(cv=5, random_state=42)
)

lasso_pipeline.fit(X_train, y_train)
lasso_model = lasso_pipeline.named_steps['lassocv']

print(f"최적 alpha: {lasso_model.alpha_}")
print(f"Train R²: {lasso_model.score(X_train, y_train):.4f}")

# 9. 예측
y_pred = lasso_pipeline.predict(test_encoded)

# 10. 제출파일 생성
submit = pd.read_csv("sample_submission.csv")
submit["SalePrice"] = y_pred
submit.to_csv("lasso_baseline.csv", index=False)

print("lasso_baseline.csv 저장 완료!")




# 펭귄데이터
#########################################


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from palmerpenguins import load_penguins
# 1. 데이터 불러오기
df = load_penguins()
df = df.dropna()
numeric_df = df.select_dtypes(include=['number'])
categorical_df = df.select_dtypes(include=['object', 'category'])
categorical_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)
df= pd.concat([numeric_df, categorical_df], axis=1)
# 2. 종속변수와 독립변수 분리
y = df['bill_length_mm']
X = df.drop(columns=['bill_length_mm'])
# 3. 범주형 변수 더미코딩
# 4. 선형회귀 모델 정의
model = LinearRegression()
# Define custom feature selector
def aic_score(estimator, X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("Model AIC:", model.aic)
    return -model.aic
# 5. SFS 실행
sfs = SFS(model,
          k_features='best',  # 또는 (1, X.shape[1])로 범위 지정 가능
          forward=True,
          scoring=aic_score,
          cv=0)
sfs = sfs.fit(X, y)
# 6. 결과 출력
print("선택된 피처 인덱스:", sfs.k_feature_idx_)
print("선택된 피처 이름:", sfs.k_feature_names_)
print("AIC score:", sfs.k_score_)
# 7. 중간 결과 보기
pd.DataFrame.from_dict(sfs.get_metric_dict()).T