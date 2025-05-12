# 의사결정나무
# 분류 혹은 예측

# 트리계열 모형은 변수 스케일링에 영향을 받지않음.
# 의사결정트리, randomforest, gbm, xgboost, lightgbm 등등


import pandas as pd
import numpy as np
train = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv')


train_X = train.drop(['grade'], axis = 1)
train_y = train['grade']

test_X = test.drop(['grade'], axis = 1)
test_y = test['grade']



# 파이프라인과 columntransformer를 사용하여 전처리와 모델링을 동시에 진행할 수 있다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline



num_columns = train_X.select_dtypes('number').columns.tolist()
cat_columns = train_X.select_dtypes('object').columns.tolist()

cat_preprocess = make_pipeline(
    #SimpleImputer(strategy="constant", fill_value="NA"),
    OneHotEncoder(handle_unknown="ignore", sparse_output=False)
)

num_preprocess = make_pipeline(
    SimpleImputer(strategy="mean"), # 결측치 평균으로 대체
    StandardScaler()                # 표준화
)

preprocess = ColumnTransformer(
    [("num", num_preprocess, num_columns),
    ("cat", cat_preprocess, cat_columns)]
)


# 의사결정트리 모델링
from sklearn.tree import DecisionTreeRegressor
full_pipe = Pipeline(
    [
        ("preprocess", preprocess),
        ("regressor", DecisionTreeRegressor())
    ]
)

# 파라미터 명칭 확인
DecisionTreeRegressor().get_params()
# max_depth: 의사결정 나무의 깊이
# min_samples_leaf: 리프노드가 되기 위한 최소 데이터의 수
# ccp_alpha: 리프노드에서 가지치기를 위한 파라미터 (알파) -> 비용 복잡도 가지치기


# GridSearchCV를 사용하여 하이퍼 파라미터 튜닝
decisiontree_param = {'regressor__ccp_alpha': np.arange(0.01, 0.3, 0.05)}
from sklearn.model_selection import GridSearchCV
decisiontree_search = GridSearchCV(estimator = full_pipe,
                                    param_grid = decisiontree_param,
                                    cv = 5,
                                    scoring = 'neg_mean_squared_error')

# 모델에 fitting
decisiontree_search.fit(train_X, train_y)
print('best 파라미터 조합 :', decisiontree_search.best_params_)
print('교차검증 MSE :', -decisiontree_search.best_score_)


# 최종적으로 테스트 데이터를 이용해서 모형 성능을 평가
from sklearn.metrics import mean_squared_error
dt_pred = decisiontree_search.predict(test_X)
print('테스트 MSE :', mean_squared_error(test_y, dt_pred))





'''
강사님코드
'''

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.head()
df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={'bill_length_mm': 'y',
                       'bill_depth_mm': 'x'})
df['y'].mean()
import matplotlib.pyplot as plt
df.plot(kind="scatter", x="x", y="y")
k=np.linspace(13, 22, 100)
plt.plot(k, np.repeat(df["y"].mean(),100), color="red")
# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
# 29.81
# x=15 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹
# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]
# 각 그룹 MSE는 얼마 인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)
# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1* n1 + mse2 * n2)/(n1+n2)
mse1 * (n1/(n1+n2)) + mse2 * (n2/(n1+n2))
29.23
29.81 - 29.23
# x = 20일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0]  # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)
(mse1* n1 + mse2 * n2)/(n1+n2)
29.73
29.81-29.73
df=df.query("x < 16.41")
# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x):
    n1=df.query(f"x < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)
    return float((mse1* n1 + mse2 * n2)/(n1+n2))
my_mse(15)
my_mse(13.71)
my_mse(14.01)
df["x"].min()
df["x"].max()
# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(13.2, 21.5, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])
result
x_values[np.argmin(result)]
# 14.01, 16.42, 19.4
import matplotlib.pyplot as plt
thr=16.409
df.plot(kind="scatter", x="x", y="y")
k=np.linspace(13, 22, 100)
k1=np.linspace(13, thr, 100)
k2=np.linspace(thr, 21, 100)
# plt.plot(k, np.repeat(df["y"].mean(),100), color="red")
plt.plot(k1, np.repeat(df.query(f"x < {thr}")["y"].mean(),100), color="blue")
plt.plot(k2, np.repeat(df.query(f"x >= {thr}")["y"].mean(),100), color="blue")


# x, y 산점도를 그리고 빨간 평행선 4개 그려주세요
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds = [14.01, 16.42, 19.4]
df