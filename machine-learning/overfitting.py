# 오버피팅과 릿지, 라쏘 모델 이해하기
# 오버피팅(Overfitting) : 모델이 학습 데이터에 너무 잘 맞춰져서 새로운 데이터에 대한 일반화 성능이 떨어지는 현상
# R2 값이 높다해서 그저 좋은게 아니라는 뜻.

# 릿지(Ridge) : L2 정규화 기법을 사용하여 모델의 복잡도를 줄이는 방법
# 라쏘(Lasso) : L1 정규화 기법을 사용하여 모델의 복잡도를 줄이는 방법
# 릿지와 라쏘는 오버피팅을 방지하기 위해 사용되는 정규화 기법입니다.
                # 모델의 가중치에 패널티를 부여하여 과적합을 방지합니다.

# 다항회귀모형
# 𝜙j(x) = x^j

import numpy as np
import pandas as pd
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=10, replace=False)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
x2 = np.linspace(0, 1, 100)
y2 = (2 * x + 3)
mydata = pd.DataFrame({'x': x, 'y': y})
mydata = mydata.sort_values('x').reset_index(drop=True)
print(mydata)


# 다항 회귀분석 예시 시각화
import matplotlib.pyplot as plt
x2 = np.linspace(0, 1, 100)
y2 = np.sin(2 * np.pi * x2)

plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'],
color='black', label='Observed')
plt.plot(x2, y2, color='red',
label='True Curve')
plt.title('Data and True Curve')
plt.legend()
plt.grid(True)
plt.show()


# 0차 다항 회귀식 (절편만 사용한 모델)
plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'],
color='black', label='Observed')
plt.plot(x2, y2, color='red',
label='True Curve')
plt.axhline(y=np.mean(mydata['y']),
color='blue',
label='Mean Model')
plt.title('0-degree Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()



# 1차 다항 회귀식 (선형 회귀 모델)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly1 = PolynomialFeatures(degree=1, include_bias=True)

x2 = np.linspace(0, 1, 100)
y2 = np.sin(2 * np.pi * x2)

X1 = poly1.fit_transform(mydata[['x']])
X1.shape
model1 = LinearRegression().fit(X1, mydata['y'])
y1_pred = model1.predict(poly1.transform(x2.reshape(-1, 1)))
                                         
plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'], color='black', label='Observed')
plt.plot(x2, y2, color='red', label='True Curve')
plt.plot(x2, y1_pred, color='blue', label='Degree 1')
plt.title('1-degree Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()




# 3차 다항 회귀식
poly2 = PolynomialFeatures(degree=3, include_bias=True)
X2 = poly2.fit_transform(mydata[['x']])
model2 = LinearRegression().fit(X2, mydata['y'])
y2_pred = model2.predict(poly2.transform(x2.reshape(-1, 1)))

plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'], color='black', label='Observed')
plt.plot(x2, y2, color='red', label='True Curve')
plt.plot(x2, y2_pred, color='blue', label='Degree 3 Fit')
plt.title('2-degree Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()




# 4차 다항 회귀식
poly4 = PolynomialFeatures(degree=4, include_bias=True)
X2 = poly4.fit_transform(mydata[['x']])
model2 = LinearRegression().fit(X2, mydata['y'])
y2_pred = model2.predict(poly4.transform(x2.reshape(-1, 1)))

plt.figure(figsize=(6, 4))
plt.scatter(mydata['x'], mydata['y'], color='black', label='Observed')
plt.plot(x2, y2, color='red', label='True Curve')
plt.plot(x2, y2_pred, color='blue', label='Degree 3 Fit')
plt.title('2-degree Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()



# 1~9 차 다항회귀식
for i in range(1, 10):
    poly_i = PolynomialFeatures(degree=i, include_bias=True)
    X2 = poly_i.fit_transform(mydata[['x']])
    model2 = LinearRegression().fit(X2, mydata['y'])
    y2_pred = model2.predict(poly_i.transform(x2.reshape(-1, 1)))

    plt.figure(figsize=(6, 4))
    plt.scatter(mydata['x'], mydata['y'], color='black', label='Observed')
    plt.plot(x2, y2, color='red', label='True Curve')
    plt.plot(x2, y2_pred, color='blue', label=f'Degree {i} Fit')
    plt.title(f'{i}-degree Polynomial Regression')
    plt.ylim(-2, 2)
    plt.legend()
    plt.grid(True)
    plt.show()


'''
Overfitting의 문제점
1. 9차 다항 회귀모델 사용시 문제점
- 회귀분석에서의 Rules of thumb은 10개 표본당 1개 독립변수!

2. 분석 및 예측에서의 문제점
- 관련 없는 변수들이 채택됨
- 학습 데이터에서 예측력이 좋지만 동일한 데이터 발생 모델에서의 관측값에 대한 예측력은 현저
히 떨어지게 됨. (model의 variance가 증가!)
'''
# train set
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=100, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data_learning = pd.DataFrame({'x': x, 'y': y})



# validation set
x_test = np.random.choice(np.arange(0, 1.05, 0.05), size=5, replace=True)
true_test_y = np.sin(2 * np.pi * x_test) + np.random.normal(0, 0.2, len(x_test))
data_test = pd.DataFrame({'x': x_test, 'y': [np.nan] * len(x_test)})


# train set -> train, vaild 나누기
from sklearn.model_selection import train_test_split
# 7: 3 비율로 나누기
train, valid = train_test_split(data_learning, test_size=0.3, random_state=1234)


# train 셋 vs. validation 셋 모델 성능비교
# 모델의 복잡도를 늘려가면서 데이터를 적합하고, valid 셋에서의 성능을 살펴보겠습니다.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

perform_train = []
perform_valid = []

for i in range(1, 21):
    poly = PolynomialFeatures(degree=i, include_bias=True)
    X_train = poly.fit_transform(train[['x']])
    X_valid = poly.transform(valid[['x']])
    model = LinearRegression().fit(X_train, train['y'])
    y_train_pred = model.predict(X_train)
    y_valid_pred = model.predict(X_valid)
    mse_train = mean_squared_error(train['y'], y_train_pred)
    mse_valid = mean_squared_error(valid['y'], y_valid_pred)
    perform_train.append(mse_train)
    perform_valid.append(mse_valid)
best_degree = np.argmin(perform_valid) + 1
print("Best polynomial degree:", best_degree)



# 모델 성능비교 시각화
# 빨간색 선을 최소로 만드는 모델을 찾는 것이 목표
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, 21), perform_train, label="Train MSE")
plt.plot(range(1, 21), perform_valid, label="Valid MSE", color='red')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Model Complexity vs MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()





# 8조
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 랜덤 시드를 고정하여 실행할 때마다 동일한 결과가 나오도록 설정
np.random.seed(2021)

# 0부터 1까지 0.05 간격으로 값을 선택해 40개의 샘플을 무작위로 선택 (중복 허용)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)

# 선택된 x값에 대해 sin 함수 적용하고, 평균 0, 표준편차 0.2인 정규분포 노이즈 추가
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))

# x와 y를 합쳐서 학습용 데이터프레임 생성
data_for_learning = pd.DataFrame({'x': x, 'y': y})

# 학습용 데이터셋을 학습(train)과 검증(valid) 세트로 7:3 비율로 분할
from sklearn.model_selection import train_test_split
train, valid = train_test_split(data_for_learning, test_size=0.3, random_state=1234)

# 사용할 모듈 import
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

i = 3  # 다항식의 차수를 설정 (ex. 3차 다항 회귀)

# 0~1 범위에서 100개의 균일한 값을 갖는 선형 배열 생성 (모델 곡선을 그릴 때 사용)
k = np.linspace(0, 1, 100)

# 참값 함수: y = sin(2πx)
sin_k = np.sin(2 * np.pi * k)

# 차수가 i인 다항 특성 생성기 정의 (include_bias=True면 절편 항 포함됨)
poly1 = PolynomialFeatures(degree=i, include_bias=True)

# 학습 데이터 x를 다항 특성으로 변환 (1, x, x^2, ..., x^i)
train_X = poly1.fit_transform(train[['x']])

# 선형 회귀 모델 학습
model1 = LinearRegression().fit(train_X, train['y'])

# 예측 곡선: k 값을 다항 특성으로 변환한 후 예측값 계산
model_line_blue = model1.predict(poly1.transform(k.reshape(-1, 1)))

# 학습 데이터에 대한 예측값
train_y_pred = model1.predict(poly1.transform(train[['x']]))

# 검증 데이터에 대한 예측값
valid_y_pred = model1.predict(poly1.transform(valid[['x']]))

# 학습 데이터의 평균제곱오차 (MSE)
mse_train = mean_squared_error(train['y'], train_y_pred)

# 검증 데이터의 평균제곱오차 (MSE)
mse_valid = mean_squared_error(valid['y'], valid_y_pred)

# 시각화를 위한 subplot 구성 (1행 2열)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 왼쪽 그래프: 학습 데이터 시각화
axes[0].scatter(train['x'], train['y'], color='black', label='Train Observed')  # 실제 학습 데이터 점
axes[0].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')  # 참 함수 (얇은 빨간 선)
axes[0].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')  # 모델 예측 곡선 (파란 선)
axes[0].text(0.05, -1.8, f'MSE: {mse_train:.4f}', fontsize=10, color='blue')  # 학습 MSE 텍스트 표시
axes[0].set_title(f'{i}-degree Polynomial Regression (Train)')  # 그래프 제목
axes[0].set_ylim((-2.0, 2.0))  # y축 범위 설정
axes[0].legend()  # 범례 표시
axes[0].grid(True)  # 그리드 표시

# 오른쪽 그래프: 검증 데이터 시각화
axes[1].scatter(valid['x'], valid['y'], color='green', label='Valid Observed')  # 실제 검증 데이터 점
axes[1].plot(k, sin_k, color='red', alpha=0.1, label='True Curve')  # 참 함수
axes[1].plot(k, model_line_blue, color='blue', label=f'Degree {i} Fit')  # 모델 예측 곡선
axes[1].text(0.05, -1.8, f'MSE: {mse_valid:.4f}', fontsize=10, color='blue')  # 검증 MSE 텍스트 표시
axes[1].set_title(f'{i}-degree Polynomial Regression (Valid)')  # 그래프 제목
axes[1].set_ylim((-2.0, 2.0))  # y축 범위 설정
axes[1].legend()  # 범례 표시
axes[1].grid(True)  # 그리드 표시

# 전체 레이아웃 정렬
plt.tight_layout()

# 그래프 표시
plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 생성
np.random.seed(2021)
x = np.random.choice(np.arange(0, 1.05, 0.05), size=40, replace=True)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, len(x))
data = pd.DataFrame({'x': x, 'y': y})

# train/valid 나누기
train, valid = train_test_split(data, test_size=0.3, random_state=1234)

# 결과 저장용 리스트
degrees = range(1, 16)
train_mse_list = []
valid_mse_list = []

# 차수별로 반복
for i in degrees:
    # 다항 특성 변환기 정의
    poly = PolynomialFeatures(degree=i, include_bias=True)

    # 데이터 변환
    X_train_poly = poly.fit_transform(train[['x']])
    X_valid_poly = poly.transform(valid[['x']])

    # 회귀 모델 학습
    model = LinearRegression().fit(X_train_poly, train['y'])

    # 예측
    y_train_pred = model.predict(X_train_poly)
    y_valid_pred = model.predict(X_valid_poly)

    # MSE 계산
    train_mse = mean_squared_error(train['y'], y_train_pred)
    valid_mse = mean_squared_error(valid['y'], y_valid_pred)

    train_mse_list.append(train_mse)
    valid_mse_list.append(valid_mse)

# 최적 차수 구하기 (valid MSE가 최소인 경우)
best_degree = degrees[np.argmin(valid_mse_list)]
best_mse = min(valid_mse_list)

print(f"최적 차수: {best_degree}차, Valid MSE: {best_mse:.4f}")

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(degrees, train_mse_list, marker='o', label='Train MSE')
plt.plot(degrees, valid_mse_list, marker='s', label='Valid MSE')
plt.axvline(best_degree, color='red', linestyle='--', label=f'Best Degree = {best_degree}')
plt.title('다항 회귀 차수별 MSE 비교')
plt.xlabel('다항식 차수 (Degree)')
plt.ylabel('MSE')
plt.ylim(-0.2, 1)
plt.xlim(0, 10)
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# bias 란?
# 내가 예측한 값과 실제 값의 차이
# bias가 크면 예측값과 실제값의 차이가 크다는 뜻
# bias가 작으면 예측값과 실제값의 차이가 작다는 뜻
# bias가 크면 오버피팅이 발생할 가능성이 높다.

# bias 와 variance 의 관계
# variance는 모델이 학습 데이터에 얼마나 민감하게 반응하는지를 나타내는 지표
# bias가 크면 variance가 작고, bias가 작으면 variance가 크다.
# bias와 variance는 서로 trade-off 관계에 있다.




# 우리가 랜덤으로 데이터셋과 validation 셋을 나누었기 때문에
# validation 셋에서의 성능이 좋지 않을 수 있다.
# 따라서, validation 셋에서의 성능을 높이기 위해서는
# 테스트를 여러번 반복해서 평균을 내는 방법을 사용할 수 있다.
# k-fold cross-validation을 사용
# 데이터를 k개로 나누어서 k-1개로 학습하고 1개로 검증하는 방법이다.
# cross-validation은 데이터를 여러번 나누어서 모델을 학습하고 검증하는 방법이다.
# cross-validation을 사용하면 모델의 성능을 더 정확하게 평가할 수 있다.

# 패널티를 이용한 모델을 사용해야 한다.
# 패널티를 이용한 모델은 릿지 회귀와 라쏘 회귀가 있다.


# 벡터 Norm 의미
# 예시: (2, -3, 7) 벡터의 L1 Norm
# L1 Norm = |2| + |-3| + |7| = 2 + 3 + 7 = 12
# L2 Norm은 벡터의 각 성분을 제곱한 후 더한 값의 제곱근
# L2 Norm = √(2^2 + (-3)^2 + 7^2) = √(4 + 9 + 49) = √62 ≈ 7.87

# L∞ Norm = max(|2|, |-3|, |7|) = 7
# L1 Norm은 벡터의 각 성분의 절댓값을 더한 값

# 라쏘 회귀직선
# 패널티항 추가
# L1 Norm을 사용하여 패널티를 부여하는 방법

# 람다가 0일 때는 일반적인 선형 회귀와 같고,
# 람다가 커질수록 패널티가 커져서 회귀계수가 0에 가까워진다.
# 따라서, 람다가 커질수록 모델이 단순해진다.

# 즉, 람다가 커질수록 모델이 단순해지면서 오버피팅을 방지할 수 있다.


# 릿지: 다중공선성이 있는경우 안정적이다.


# 라쏘회귀  어떻게구현하나요
# scikit-learn의 Lasso 클래스를 사용하여 라쏘 회귀를 구현할 수 있습니다.


# 예제 데이터 불러오기
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
iris = pd.DataFrame(iris.data, columns=iris.feature_names)
iris.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

X = iris[['Petal_Width', 'Sepal_Length', 'Sepal_Width']]
# X = np.column_stack((np.ones(len(X)), X)) # 절편항 추가
y = iris['Petal_Length'].values


# 라쏘 모델 적합하기
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=10)  # alpha는 패널티 항의 계수 (람다)
lasso_model.fit(X, y)
lasso_model.coef_  # 회귀계수
lasso_model.intercept_  # 절편

lasso_model = Lasso(alpha=0)  # alpha는 패널티 항의 계수 (람다)
lasso_model.fit(X, y)
lasso_model.coef_  # 회귀계수
lasso_model.intercept_  # 절편
# 람다 값이 커지면 회귀계수가 0에 가까워진다.
# 람다 값이 작아지면 회귀계수가 커진다.
# 람다를 크게 설정하면 죽는 변수 개수가 많아진다.
# 람다가 작으면 변수를 많이 사용하게 된다.


# 릿지로 바꾸자
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1.1)  # alpha는 패널티 항의 계수 (람다)
ridge_model.fit(X, y)
ridge_model.coef_  # 회귀계수
ridge_model.intercept_  # 절편

# 릿지회귀는 람다가 커도 계수가 0이되지 않는다.
# 단, 베타 계수 벡터의 L2 Norm 값이 작아짐.
# 릿지를 적용하면 좋은 이유: 다중공선성 때문에 계수가 커지는 것을 방지할 수 있다.
# house price 데이터 최적 람다 찾아보기


# ElasticNet
# 릿지와 라쏘를 혼합한 모델
# L1 Norm과 L2 Norm을 모두 사용하여 패널티를 부여하는 방법
from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)  # alpha는 패널티 가중치, l1_ratio는 L1 Norm과 L2 Norm의 비율
elastic_model.fit(X, y)
elastic_model.coef_
elastic_model.intercept_


# CV 를 통한 람다 찾기 - 라쏘
import pandas as pd
import numpy as np
data_X = pd.read_csv("./QuickStartExample_x.csv")
y = pd.read_csv("./QuickStartExample_y.csv")


# 람다 0.5인경우
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.5)  # alpha는 패널티 항의 계수 (람다)
lasso_model.fit(data_X, y)
lasso_model.coef_  # 회귀계수
lasso_model.intercept_  # 절편


# data_X 를 train / valid 나누기
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(data_X, 
                                                      y, 
                                                      test_size=0.3, 
                                                      random_state=2025)

# 예시: 람다 0.1씩 늘려서 예측 성능 평가
for i in range(1, 6):
    lasso_model = Lasso(alpha=i / 10)  # alpha는 패널티 항의 계수 (람다)
    lasso_model.fit(X_train, y_train)

    # Validation Set 에서의 성능 평가
    y_valid_hat = lasso_model.predict(X_valid)    # 예측 y값

    # MSE 어떻게 계산?
    print(f"MSE (lambda-{i / 10}): ", sum((y_valid_hat - y_valid['V1'])** 2))



# 최적 람다 찾기
alphas = np.linspace(0, 0.5, 1000)
valid_mse = []
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_valid_hat = model.predict(X_valid)    # 예측 y값
    valid_mse.append(sum((y_valid_hat - y_valid['V1'])** 2))

best_alpha = alphas[np.argmin(valid_mse)]
print(f"Best alpha(람다): {best_alpha}")


import matplotlib.pyplot as plt

# 최적 람다 시각화
plt.figure(figsize=(8, 5))
plt.scatter(alphas, valid_mse, color='blue', label='Validation MSE')
plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best Alpha = {best_alpha:.2f}')
plt.legend()
plt.xlabel('Alpha (람다)')
plt.ylabel('MSE')
plt.show()


# 최적 람다를 통한 라쏘회귀
lasso_model = Lasso(alpha=best_alpha)
lasso_model.fit(X_train, y_train)
lasso_model.coef_  # 회귀계수
lasso_model.intercept_  # 절편


# Cross Validation
from sklearn.linear_model import LassoCV
alphas = np.linspace(0, 0.5, 1000)
lasso_cv = LassoCV(alphas=alphas, 
                   cv=5,            # 5-fold cross-validation
                   max_iter=10000)


# 데이터 학습 (train set, valid set 나누지 않아도됨)
lasso_cv.fit(data_X, y)             
lasso_cv.alpha_      # 최적 람다
lasso_cv.mse_path_.shape    # 1000, 5 (람다 개수, fold 개수)
lasso_cv.mse_path_[:, 0]    # 첫번째 fold의 mse








'''
금요일 프로젝트 관련 공지 입니다.

- 데이터: Ames 데이터(lon, lat 버전) + 외부데이터(자유)
- 주제: Ames 데이터와 관련한 자유주제 – 예시는 지난 3기 자료 참고 할 것.
- 형식: 대쉬보드 (스태틱) + 인터렉티브 시각화

[필수 요구사항]
1. 데이터 탐색(EDA) 및 전처리 결과 시각화
- 주요 변수 분포, 결측치 처리, 이상치 탐지 등

2. 지도 기반 시각화
- 예: Folium, Plotly 등 사용 가능

3. 인터랙티브 요소
- 예: Plotly 등

4. 모델 학습 페이지
- 회귀모델 훈련 과정과 결과 시각화
- 페널티 회귀 모델 필수 사용

5. 스토리텔링 구성
- 전체 대시보드가 하나의 분석 흐름으로 자연스럽게 이어질 것
- 꼭 집값 예측이 아니어도 됨!

6. 전체 분량
- 4-5페이지로 구성


3기 자료 참고
지난 기수 Ames 관련 발표 대시보드 참고용

https://h-yoeunk.github.io/testdashboard/
https://yongraegod.github.io/Ames_Project/#
https://otacute.github.io/whaleshark/#
https://bonboneee.github.io/Project2/#
https://summerdklee.github.io/mywebsite/posts/team_proj_2/
https://ohseyn.github.io/00/#

'''