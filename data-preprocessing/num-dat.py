# 데이터 전처리
# 4. 수치형 변수 이산화 방법
import numpy as np
import pandas as pd
X = np.array([[0, 1, 1, 2, 5, 10, 11, 14, 18]]).T

# 구간의 길이가 같도록 이산화하는 방법
from sklearn.preprocessing import KBinsDiscretizer
kbd = KBinsDiscretizer(n_bins = 3, strategy = 'uniform') # 구간의 길이가 동일

X_bin = kbd.fit_transform(X).toarray()
print(kbd.bin_edges_)

# 사분위수를 기준으로 이산화
kbd2 = KBinsDiscretizer(n_bins = 4,
                        strategy = 'quantile') # 사분위수를 기준으로 이산화
X_bin2 = kbd2.fit_transform(X).toarray()
print(kbd2.bin_edges_)


# 구간을 임의로 설정하는 방법
bins = [0, 4, 7, 11, 18]
labels = ['A', 'B', 'C', 'D']
X_bin3 = pd.cut(X.reshape(-1),
                bins = bins,
                labels = labels)
print(X_bin3)


# 범주형 변수 축소 방법
# 빈도수가 낮은 범주를 하나의 범주로 통합하는 방법
train_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_train.csv')
test_bike = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_test.csv')

print(train_bike.head(2))

# 빈도 확인하기
train_bike['weather'].value_counts()

# 상대 비율 확인
freq = train_bike['weather'].value_counts(normalize = True)
print(freq)

prob_columns = train_bike['weather'].map(freq)
prob_columns.head(2)

# 상대 비율이 0.1보다 작은 경우 'other'로 대치
# mask(조건, 마스킹 값): 조건을 만족하는 데이터를 특정 값으로 마스킹
train_bike['weather'] = train_bike['weather'].mask(prob_columns < 0.1, 'other')
test_bike['weather'] = np.where(test_bike['weather'].isin([4]), 'other', test_bike['weather'])

print(train_bike['weather'].value_counts())