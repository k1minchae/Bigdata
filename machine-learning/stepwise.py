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

# sfs = SFS(lr,                        # 분류기나 회귀기
#           k_features=5,              # 최종 선택할 변수 개수
#           forward=True,              # forward 방식
#           floating=False,            # floating 방식 사용 여부
#           scoring='r2',              # 평가 지표 (회귀 → R², 분류 → accuracy 등)
#           cv=5)                      # 교차검증 folds 수
# fixed_features=[0, 1] 고정할 변수의 인덱스

# sfs = sfs.fit(X, y)

print("선택된 변수:", sfs.k_feature_names_)


print('Selected features:', np.array(names)[list(sfs.k_feature_idx_)])
# Model AIC: 385.1349961458542
# Model AIC: 568.7536581517621
# Model AIC: 206.35386357991942
# Model AIC: 156.17922963502616
# Model AIC: 193.99401214096014
# Model AIC: 86.81602018114887  (작을수록 좋기 때문에 이게 젤 좋다.)

# SFS



# 타깃 변수: y (SalePrice와 같은 개념)
y = np.array([78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7,
              72.5, 93.1, 115.9, 83.8, 113.3, 109.4])

# 독립 변수 4개 (설명 변수 집합)
X = np.array([[17, 26, 6, 60],
              [1, 29, 15, 52],
              [11, 56, 8, 20],
              [11, 31, 8, 47],
              [7, 52, 6, 33],
              [11, 55, 9, 22],
              [3, 71, 17, 6],
              [1, 31, 22, 44],
              [2, 54, 18, 22],
              [21, 47, 4, 26],
              [1, 40, 23, 34],
              [11, 66, 9, 12],
              [10, 68, 8, 12]])



from sklearn.metrics import r2_score

def adjusted_r2_score(estimator, X, y):
    y_pred = estimator.predict(X)         # 예측값 계산
    n = X.shape[0]                        # 샘플 수 (관측값 수)
    p = X.shape[1]                        # 독립 변수 개수
    r2 = r2_score(y, y_pred)              # 일반 R² 계산
    # 수정된 R² 공식 (모델에 사용된 변수 수 고려)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2                   # 조정된 R² 반환

# 사용할 회귀 모델 정의
lr = LinearRegression()

# 순차적 특성 선택기 정의
sfs = SFS(estimator=lr,                # 사용할 모델 (선형 회귀)
          k_features=(1, 4),           # 1~4개 변수 중 최적 조합 탐색
          forward=True,                # Forward 방식: 하나씩 추가하며 선택
          scoring=adjusted_r2_score,   # 사용자 정의 평가 함수 사용 (조정 R²)
          cv=0,                        # 교차검증 사용 안 함 (그대로 학습)
          verbose=2)                   # 출력 생략

sfs.fit(X, y)       # # 실제 특성 선택 수행

print("선택된 특성 인덱스:", sfs.k_feature_idx_)
print("선택된 특성 이름:", sfs.k_feature_names_)
print("조정된 R²:", sfs.k_score_)