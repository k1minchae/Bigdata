# 데이터 전처리 기법
# 1. 결측치 처리
import pandas as pd
import numpy as np

dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/dat.csv')

y = dat.grade
X = dat.drop(['grade'], axis = 1)

# 각 칼럼별 속성 확인
print(X.info())
print(y.info())

# 결측치 확인
print(dat.isna().sum(axis = 0))

# 학습 데이터와 테스트 데이터 분리
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(
                    X,
                    y,
                    test_size = 0.2,
                    random_state = 0,
                    shuffle = True,
                    stratify = None
)


# 결측치 처리
# fit_transform() : 학습 데이터에 적용 => 훈련데이터의 통계량을 이용하여 결측치 대치
# transform() : 테스트 데이터에 적용 => 훈련데이터의 통계량을 이용하여 결측치 대치
# 테스트데이터의 정보를 활용하면 안 됨
# ★ 데이터 누수: 학습 데이터에 포함된 정보가 테스트 데이터에 포함되는 것

# 1. 통계량을 이용한 대치
# 평균대치법: 결측치의 개수가 적을 때 빠르게 대치 가능해서 좋다.
from sklearn.impute import SimpleImputer
train_X1 = train_X.copy()
test_X1 = test_X.copy()

# mean으로 대치
imputer_mean = SimpleImputer(strategy = 'mean')

# 데이터에 적용
train_X1['goout'] = imputer_mean.fit_transform(train_X1[['goout']])
test_X1['goout'] = imputer_mean.transform(test_X1[['goout']])

print('학습 데이터 goout 변수 결측치 확인 :', train_X1['goout'].isna().sum())
print('테스트 데이터 goout 변수 결측치 확인 :', test_X1['goout'].isna().sum())


# 중앙값 대치법
train_X2 = train_X.copy()
test_X2 = test_X.copy()

imputer_median = SimpleImputer(strategy = 'median')
train_X2['goout'] = imputer_median.fit_transform(train_X2[['goout']])
test_X2['goout'] = imputer_median.transform(test_X2[['goout']])

print('학습 데이터 goout 변수 결측치 확인 :', train_X2['goout'].isna().sum())
print('테스트 데이터 goout 변수 결측치 확인 :', test_X2['goout'].isna().sum())


# 최빈값 대치법 => 범주형 변수에 적용
train_X3 = train_X.copy()
test_X3 = test_X.copy()

imputer_mode = SimpleImputer(strategy = 'most_frequent')
train_X3['goout'] = imputer_mode.fit_transform(train_X3[['goout']])
test_X3['goout'] = imputer_mode.transform(test_X3[['goout']])

print('학습 데이터 goout 변수 결측치 확인 :', train_X3['goout'].isna().sum())
print('테스트 데이터 goout 변수 결측치 확인 :', test_X3['goout'].isna().sum())



##################################################################################ㄴ
# 2. KNN을 이용한 대치
# KNN 모델을 활용하여 k개의 이웃을 택한 후, 이웃 관측치의 정보를 활용하여 결측치를 대치
# 데이터에 대한 가정 없이 쉽고 빠르게 결측치 대치 가능
# 단점: 변수 스케일 및 이상치에 민감하며, 
#      고차원 데이터의 경우 KNN 모델 성능이 떨어질 수 있음

from sklearn.impute import KNNImputer
train_X5 = train_X.copy()
test_X5 = test_X.copy()

# 수치형 변수와 범주형 변수를 나누어 대치
train_X5_num = train_X5.select_dtypes('number')
test_X5_num = test_X5.select_dtypes('number')
train_X5_cat = train_X5.select_dtypes('object')
test_X5_cat = test_X5.select_dtypes('object')

# KNNImputer는 수치형 변수에만 적용 가능
# k값에 따라서 대치 결과가 달라질 수 있음 => k값을 조정하여 성능을 높일 수 있음
knnimputer = KNNImputer(n_neighbors = 5)

train_X5_num_imputed = knnimputer.fit_transform(train_X5_num)
test_X5_num_imputed = knnimputer.transform(test_X5_num)

# sklearn은 numpy로 output이 나오기 때문에 DataFrame으로 변환해줘야 한다.
train_X5_num_imputed = pd.DataFrame(train_X5_num_imputed,
                                    columns=train_X5_num.columns,
                                    index = train_X5.index)
test_X5_num_imputed = pd.DataFrame(test_X5_num_imputed,
                                    columns=test_X5_num.columns,
                                    index = test_X5.index)

train_X5 = pd.concat([train_X5_cat, train_X5_num_imputed], axis = 1)
test_X5 = pd.concat([test_X5_cat, test_X5_num_imputed], axis = 1)

print('학습 데이터 goout 변수 결측치 확인 :', train_X5['goout'].isna().sum())
print('테스트 데이터 goout 변수 결측치 확인 :', test_X5['goout'].isna().sum())

# set_output() 메서드를 활용하면 추가적인 코드 작성 없이 pandas 데이터프레임으로 변환할 수 있습니다.
knnimputer2 = KNNImputer(n_neighbors = 5).set_output(transform = 'pandas')
train_X5_num_imputed2 = knnimputer2.fit_transform(train_X5_num)
test_X5_num_imputed2 = knnimputer2.transform(test_X5_num)

# 판다스 데이터프레임 출력
print(train_X5_num_imputed2.head())
print('학습 데이터 goout 변수 결측치 확인 :', train_X5_num_imputed2['goout'].isna().sum())
print('테스트 데이터 goout 변수 결측치 확인 :', test_X5_num_imputed2['goout'].isna().sum())

# Sklearn 을 왜 쓰는걸까?
# 데이터 누수 방지에 용이하다.


########################################################################################
# 판다스를 활용하여 결측치를 대치하는 방법
# 주로 전체 데이터를 대치할 때 사용

data = {
    '학생': ['철수', '영희', '민수', '수지', '지현'],
    '수학': [85, np.nan, 78, np.nan, 93],
    '영어': [np.nan, 88, 79, 85, np.nan],
    '과학': [92, 85, np.nan, 80, 88]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

df1 = df.copy()
df1['수학'].fillna(df1['수학'].mean(), inplace=True)
df1['영어'].fillna(df1['영어'].mean(), inplace=True)
df1['과학'].fillna(df1['과학'].mean(), inplace=True)
print(df1)

