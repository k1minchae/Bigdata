# 데이터 전처리
# 부록: 데이터 통합
# make_column_transformer() 를 통해 각 컬럼별로 전처리 방법을 지정할 수 있음
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer

dat = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bda1.csv')
y = dat.grade
X = dat.drop(['grade'], axis = 1)
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

cat_columns = train_X.select_dtypes('object').columns
num_columns = train_X.select_dtypes('number').columns


# 범주형 변수에는 원-핫 인코딩, 수치형 변수에는 표준화
from sklearn.preprocessing import OneHotEncoder, StandardScaler
onehotencoder = OneHotEncoder(sparse_output = False,    # 연산 속도 빨라짐
                            drop = None,
                            handle_unknown = 'ignore')

stdscaler = StandardScaler()    # 표준화 모듈 불러오기

mc_transformer = make_column_transformer(
        (onehotencoder, cat_columns),   # 범주형 변수: 원-핫 인코딩
        (stdscaler, num_columns),       # 수치형 변수: 표준화
        remainder='passthrough'         # 나머지 변수는 그대로 유지
        ).set_output(transform = 'pandas')

train_X_transformed = mc_transformer.fit_transform(train_X)
test_X_transformed = mc_transformer.transform(test_X)

print(train_X_transformed.head(2))

# 장점: 각 전처리 방법을 일관되게 적용할 수 있음
# 단점: 개별적으로 전처리한 것과 비교했을 때, 각 전처리 방법의 결과를 확인하기 어려움