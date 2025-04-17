# AIC Base stepwise method
# AIC, BIC는 무엇일까?
# AIC
# 모델에 사용되는 변수는 적으면 적을수록 좋다.
# AIC는 모델의 적합도와 복잡도를 모두 고려하여 모델을 평가하는 방법이다.
# AIC는 모델의 적합도가 높을수록 낮아지며, 모델의 복잡도가 높을수록 높아진다.
# p: 모델에 사용되는 변수 개수
# 지표가 낮을수록 좋다.

# AIC = -2 * log-likelihood + 2 * p
# BIC
# BIC는 AIC와 비슷하지만, 모델의 복잡도에 대한 패널티가 더 크다.


"""Step wise 방법 : 단계별로 변수를 추가해나가면서 최적의 모델을 찾는 방법.
보통은 추가할거랑, 제거할거를 p-value 값을 0.15를 기준으로 선택함

- 자동으로 베스트 모델을 찾아줌.
- R square 값이 자동으로 높게 측정되는 경향성을 띈다.
- 신뢰구간이 비정상적으로 높게나오는 구간 선택됨
- 다중공선성이 문제가 되는 데이터의 경우 잘 작동하지 않는다.

AIC, BIC 
Base stepwise method"""


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from sklearn.datasets import load_iris


# Load iris data
iris = load_iris()
X = iris.data[:, [0, 1, 3]]
y = iris.data[:, 2]
names = np.array(iris.feature_names)[[0, 1, 3]]

# Define model
lr = LinearRegression()

# Define custom feature selector
def aic_score(estimator, X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print("Model AIC:", model.aic)
    return -model.aic

# Perform SFS
sfs = SFS(lr,
    k_features=(1,3),
    forward=True, # 방향에 대한 옵션
    scoring=aic_score,
    cv=0, 
    verbose = False)
sfs.fit(X, y)

print('Selected features:', np.array(names)[list(sfs.k_feature_idx_)])
# Model AIC: 385.1349961458542
# Model AIC: 568.7536581517621
# Model AIC: 206.35386357991942
# Model AIC: 156.17922963502616
# Model AIC: 193.99401214096014
# Model AIC: 86.81602018114887  (작을수록 좋기 때문에 이게 젤 좋다.)

