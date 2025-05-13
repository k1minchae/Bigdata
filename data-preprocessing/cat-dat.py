# 데이터 전처리 기법
# 2. 범주형 변수 인코딩
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

###################################################################################
# 범주형 변수 인코딩

# 1. 라벨 인코딩
# 범주형 변수의 각 label에 알파벳 순서대로 고유한 정수를 할당하는 방법
# 순서형 변수의 경우 순서를 반영한 인코딩이 가능함
# labelencoder = LabelEncoder() 1차원으로 출력: Series 형식 (주로 종속변수)
from sklearn.preprocessing import OrdinalEncoder # 2차원으로 출력: DataFrame 형식

train_X6 = train_X.copy()
test_X6 = test_X.copy()

train_X6_cat = train_X6.select_dtypes('object')
test_X6_cat = test_X6.select_dtypes('object')

ordinalencoder = OrdinalEncoder().set_output(transform = 'pandas')
train_X6_cat = ordinalencoder.fit_transform(train_X6_cat)
test_X6_cat = ordinalencoder.fit_transform(test_X6_cat)

print(train_X6_cat.head(2))


# 테스트 데이터에는 있지만 훈련 데이터에는 없는 범주가 있을 수 있음 !!
# 훈련 데이터
train_data = pd.DataFrame({
        'job': ['Doctor', 'Engineer', 'Teacher', 'Nurse']
})

# 테스트 데이터
test_data = pd.DataFrame({
        'job': ['Doctor', 'Lawyer', 'Teacher', 'Scientist']
})

# 학습되지 않은 카테고리를 처리할 때 오류를 발생시키지 않고 unknown_value로 대치
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# 훈련 데이터로 인코더 학습
oe.fit(train_data[['job']])

# 훈련 데이터 변환
train_data['job_encoded'] = oe.transform(train_data[['job']])

# 테스트 데이터 변환 (훈련 데이터에 없는 직업은 -1로 인코딩됨)
test_data['job_encoded'] = oe.transform(test_data[['job']])
print(train_data)
print(test_data)


###################################################################################
# 2. 원핫 인코딩 : 범주형 변수만큼 새로운 열 생성 (모두 포함)
# 장점: 라벨 인코딩의 문제점인 수치 정보 반영 문제 해결 가능
# 단점: 차원 증가로 인한 메모리 사용량 증가
from sklearn.preprocessing import OneHotEncoder

train_X7 = train_X.copy()
test_X7 = test_X.copy()

train_X7_cat = train_X7.select_dtypes('object')
test_X7_cat = test_X7.select_dtypes('object')

onehotencoder = OneHotEncoder(sparse_output = False,
                            handle_unknown = 'ignore').set_output(transform = 'pandas')

train_X7_cat = onehotencoder.fit_transform(train_X7_cat)
test_X7_cat = onehotencoder.transform(test_X7_cat)

print(train_X7_cat.head())
print(test_X7_cat.head())


# 3. 더미 인코딩 : 기준범주를 제외한 나머지 범주에 대해서만 더미변수를 생성
train_X8 = train_X.copy()
test_X8 = test_X.copy()

train_X8_cat = train_X8.select_dtypes('object')
test_X8_cat = test_X8.select_dtypes('object')

# drop = 'first' : 첫 번째 범주를 기준범주로 설정
# handle_unknown = 'error' : 훈련 데이터에 없는 범주가 테스트 데이터에 있는 경우 오류 발생
dummyencoder = OneHotEncoder(sparse_output = False,
                        drop = 'first',
                        handle_unknown = 'error').set_output(transform = 'pandas')

train_X8_cat = dummyencoder.fit_transform(train_X8_cat)
test_X8_cat = dummyencoder.transform(test_X8_cat)

print(train_X8_cat.head())
