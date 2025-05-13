# 데이터 전처리 기법
# 3. 데이터 정규화

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 실습 데이터 불러오기
bike_data = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv")

####################################################################################

# 분포의 치우침이 있을 때, 변수 변환을 통해 정규분포 형태로 변환하는 것을 고려
# box-cox 변환, Yeo-Johnson 변환
# box-cox 변환은 데이터의 범위가 0보다 큰 경우에 한해서 적용이 가능
# Yeo-Johnson 변환은 데이터의 범위에 상관없이 적용 가능
# 해석을 하려면 다시 원데이터로 변환해야 한다.

from sklearn.preprocessing import PowerTransformer
import warnings
np.warnings = warnings

bike_data['count'].hist();
plt.show();
# 우측꼬리분포이다.

# box-cox 변환
box_tr = PowerTransformer(method = 'box-cox')
bike_data['count_boxcox'] = box_tr.fit_transform(
bike_data[['count']])
print('lambda : ', box_tr.lambdas_)

# box-cox 변환 후 데이터 분포 확인
bike_data['count_boxcox'].hist();
plt.show()


###################################################################################
# 데이터 불러오기

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


# 정규화의 장점
# - 모델 학습시 모형의 수렴 속도 향상
# - 거리 계산시 변수별 스케일 통일 => 모든 변수가 동등하게 기여

# 1. 표준화
# - 데이터의 평균을 0, 표준편차를 1로 변환
# - 변수의 스케일을 통일 시켜주는 거지 분포의 형태를 바꾸지는 않음
# 단점: 이상치에 민감함
from sklearn.preprocessing import StandardScaler

train_X9 = train_X.copy()
test_X9 = test_X.copy()

train_X9_num = train_X9.select_dtypes('number')
test_X9_num = test_X9.select_dtypes('number')

stdscaler = StandardScaler().set_output(transform = 'pandas')
train_X9_num = stdscaler.fit_transform(train_X9_num)
test_X9_num = stdscaler.transform(test_X9_num)
print(train_X9_num.head(2))

print('변환 전 평균 :', np.round(train_X9['absences'].mean()), sep = '\n')
print('변환 후 평균 :', np.round(train_X9_num['absences'].mean()), sep = '\n')


# 변환전후 분포비교
fig, axs = plt.subplots(nrows=1, ncols=2)
train_X9['absences'].hist(ax=axs[0], color='blue', alpha=0.7)
axs[0].set_title('before transformation')

train_X9_num['absences'].hist(ax=axs[1], color='red', alpha=0.7)
axs[1].set_title('after transformation')
plt.tight_layout();
plt.show();




# 2. Min-Max 정규화
# - 데이터의 최소값과 최대값을 이용하여 0~1 사이로 변환
# 단점: 이상치가 있을 경우 값이 너무 작아지거나 커질 수 있음
from sklearn.preprocessing import MinMaxScaler

train_X10 = train_X.copy()
test_X10 = test_X.copy()

train_X10_num = train_X10.select_dtypes('number')
test_X10_num = test_X10.select_dtypes('number')

minmaxscaler = MinMaxScaler().set_output(transform = 'pandas')
train_X10_num = minmaxscaler.fit_transform(train_X10_num)
test_X10_num = minmaxscaler.transform(test_X10_num)

range_df = train_X10_num.select_dtypes('number').apply(lambda x: x.max() - x.min(), axis=0)
print("\nRange of each column:")
print(range_df)


# 히스토그램 그려보고 각 구간별 빈도값 확인하고 이상치 확인해서 적절한거 쓰자
