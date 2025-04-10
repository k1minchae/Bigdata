# 공분산
# convariance(x, y)
# 팔길이:x, 다리길이:y
# 변수들이 하나씩 관찰되는 것이 아니라 동시에 관찰되는 경우.

# x열 y행
# arr = [[1/6, 1/3, 1/12],
#         [2/9, 1/6, 0],
#         [1/36, 0, 0]]
# e_x = 0*15/36 + 1 * 12 + 2 * 1/12 = 2/3
# e_y = 0 * 7/12 + 1 * 7/18 + 2 * 1/36 = 4/9
# e[x|y = 1] = ?

# 조건부기대값




'''
4/9
'''
import numpy as np

# 설정: 표본 크기
n = 1000

# 두 변수의 평균
mean = [3, 4]

# 공분산 행렬 => x1분산, x2분산 우상 대각선
# (0, 1), (1, 0) 은 항상 같다. (cov = 12)
cov = [[2**2, 3.6],
       [3.6, 3**2]]

# 다변량 정규분포로부터 난수 추출
np.random.seed(42)
data = np.random.multivariate_normal(mean, cov, size=n)

x1 = data[:, 0]
x2 = data[:, 1]

import matplotlib.pyplot as plt

# 문제
# 각 X1, X2 의 표뵨평균과 표본 분산
# 상관계수까지 그리기

# 표본 평균과 분산 계산
x1_bar = np.mean(x1)
x2_bar = np.mean(x2)
var_x1 = np.var(x1, ddof=1)  
var_x2 = np.var(x2, ddof=1)

# 표본 상관계수 계산
corr = np.corrcoef(x1, x2)[0, 1]
# 표본 공분산 계산
x12_cov = np.cov(x1, x2)[0, 1]
sum((x1 - x1_bar) * (x2 - x2_bar)) / (n - 1) # 계산식

# 결과 출력
print(f"X1의 표본 평균: {x1_bar:.2f}, 표본 분산: {var_x1:.2f}")
print(f"X2의 표본 평균: {x2_bar:.2f}, 표본 분산: {var_x2:.2f}")
print(f"X1과 X2의 상관계수: {corr:.2f}")
print(f"표본 공분산: {x12_cov:.2f}")

# 히스토그램 시각화
plt.figure(figsize=(10, 4))
plt.hist(x1, bins=30, alpha=0.7, label='X1')
plt.hist(x2, bins=30, alpha=0.7, label='X2')
plt.title('X1, X2 분포 히스토그램')
plt.xlabel('값')
plt.ylabel('빈도')
plt.legend()
plt.grid(True)
plt.show()


upper_r = sum((x1 - x1_bar) * (x2 - x2_bar))
lower_r = np.sqrt(sum(x1 - x1_bar)) * np.sqrt(sum(x2 - x2_bar))
r = upper_r / lower_r
r

np.cov(x1, x2, ddof=1)
np.corrcoef(x1, x2)

plt.scatter(x1, x2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

import scipy.stats as stats
corr_coef, p_val = stats.pearsonr(x1, x2)
p_val < 0.05    # True (상관관계가 있다.)


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris 
# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

# 1. Iris 데이터 로드
df_iris = load_iris() 

# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시
# 3. 타겟(클래스) 추가
iris["species"] = df_iris.target
# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["species"] = iris["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

iris["species"].unique()


# 회귀분석
import statsmodels.api as sm
import statsmodels.formula.api as smf

mode1l = smf.ols("Petal_Length ~ Petal_Width + Sepal_Width", # 종속변수 ~ 독립변수1 + 독립변수2
                data=iris).fit()

print(model.summary())
# Dep Variable: 종속변수
# OLS: Ordinary Least Square (y1- y1^)의 제곱합을 최소로 만드는
# 인터셉트: 절편
# coef / std err => t통계량
# t: 0이아니다 라는 귀무가설에 대한 t값과 p > |t|: p-value 값
# f검정: 해당 회귀모델이 유효한지 알려줌
# F값이 크면? -> 좋다.

x1 = iris["Petal_Width"]
x2 = iris["Sepal_Width"]
y = iris["Petal_Length"]

b1 = model.params["Petal_Width"]
b2 = model.params["Sepal_Width"]
b0 = model.params["Intercept"]

x2_mean = x2.mean()
x1_mean = x1.mean()

# x축 범위에 맞춰 선형 간격
x_line = np.linspace(x1.min(), x1.max(), 100)  

# y = ax + b
y_line = b1 * x_line + b0

x_line1 = np.linspace(x1.min(), x1.max(), 100)
y_line1 = b0 + b1 * x_line1 + b2 * x2_mean  # x2는 평균으로 고정

plt.figure(figsize=(6, 4))
plt.scatter(x1, y, alpha=0.7)
plt.plot(x_line1, y_line1, color='red', label='회귀직선 (x2 평균 고정)')
plt.xlabel("꽃잎 너비")
plt.ylabel("꽃잎 길이")
plt.title("꽃잎 너비 vs 꽃잎 길이")
plt.legend()
plt.grid(True)
plt.show()



x_line2 = np.linspace(x2.min(), x2.max(), 100)
y_line2 = b0 + b1 * x1_mean + b2 * x_line2  # x1은 평균으로 고정

plt.figure(figsize=(6, 4))
plt.scatter(x2, y, alpha=0.7)
plt.plot(x_line2, y_line2, color='blue', label='회귀직선 (x1 평균 고정)')
plt.xlabel("꽃받침 너비")
plt.ylabel("꽃잎 길이")
plt.title("꽃받침 너비 vs 꽃잎 길이")
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

x1 = iris["Petal_Width"]
y = iris["Petal_Length"]

b1 = model.params["Petal_Width"]
b0 = model.params["Intercept"]

# x축 범위에 맞춰 선형 간격
x_line = np.linspace(x1.min(), x1.max(), 100)  

# y = ax + b
y_line = b1 * x_line + b0

x_line1 = np.linspace(x1.min(), x1.max(), 100)
y_line1 = b0 + b1 * x_line1 + b2

plt.figure(figsize=(6, 4))
plt.scatter(x1, y, alpha=0.7)
plt.plot(x_line1, y_line1, color='red', label='회귀직선 (x2 평균 고정)')
plt.xlabel("꽃잎 너비")
plt.ylabel("꽃잎 길이")
plt.title("꽃잎 너비 vs 꽃잎 길이")
plt.legend()
plt.grid(True)
plt.show()



# 다중회귀 평면도 그리기
import matplotlib.pyplot as plt
import numpy as np

# 독립변수
x1 = iris["Petal_Width"]
x2 = iris["Sepal_Width"]
y = iris["Petal_Length"]

# 회귀 계수
b0 = model.params["Intercept"]
b1 = model.params["Petal_Width"]
b2 = model.params["Sepal_Width"]

# 3D figure 생성
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 산점도 찍기
ax.scatter(x1, x2, y, color='blue', alpha=0.6, label='관측값')

# 그리드 만들기
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 30),
    np.linspace(x2.min(), x2.max(), 30)
)

# 회귀 평면
y_pred = b0 + b1 * x1_grid + b2 * x2_grid
ax.plot_surface(x1_grid, x2_grid, y_pred, alpha=0.3, color="red", edgecolor='none', label='회귀평면')

# 축 라벨
ax.set_xlabel("꽃잎 너비 (Petal_Width)")
ax.set_ylabel("꽃받침 너비 (Sepal_Width)")
ax.set_zlabel("꽃잎 길이 (Petal_Length)")
ax.set_title("다중 회귀 평면 시각화")

plt.tight_layout()
plt.show()



# 회귀직선 기울기, 절편의 추정값에 대해서,
# 잔차 제곱합을 최소로만드는 B를 찾아야한다.
# 변수 3개 모델 베타 벡터 구해보세요.
y = iris["Petal_Length"]
x1 = iris["Petal_Width"]
x2 = iris["Sepal_Width"]
x3 = iris["Sepal_Length"]


model2 = smf.ols("Petal_Length ~ Petal_Width + Sepal_Width + Sepal_Length", # 종속변수 ~ 독립변수1 + 독립변수2
                data=iris).fit()


data_X = np.column_stack([np.repeat(1, 150, )])

# 4. 베타 벡터 출력
beta_vector = model.params.values
print("베타 벡터:", beta_vector)
print(model.summary())



import numpy as np

# 1. 종속변수
y = iris["Petal_Length"].values  # numpy 배열로 변환

# 2. 독립변수: 상수항 + 3개의 변수 (X 행렬 만들기)
x1 = iris["Petal_Width"].values
x2 = iris["Sepal_Width"].values
x3 = iris["Sepal_Length"].values

# 3. column_stack 으로 하나로 묶기
X = np.column_stack([np.ones(len(x1)), x1, x2, x3])  # np.ones는 절편항 추가

# 4. 베타 벡터 계산 (정규 방정식 사용)
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# 5. 결과 출력
print(beta)



# F검정으로 모델 간 비교
# 두 모델의 오차 제곱합 비교
# 1변수 오차 - 3변수 오차
# 성능이좋을수록 F값이 커진다.

table = sm.stats.anova_lm(mode1l, model2)   # anova
print(table)
# 귀무가설: 모델1, 2 가 성능이 같다. (즉, 추가된 변수는 쓸모 없다)
# 대립가설: 더 복잡한 모델은 기존 모델보다 설명력이 유의하게 증가했다.
# → 모델 2는 모델 1보다 유의하게 좋다
# → 추가된 변수는 의미 있는 설명력을 제공한다



# 회귀 직선계수를 벡터 형식으로 표현하는방법
# B_hat = (XT * X)^-1 * (XT * Y)

# 회귀분석표 해석
# 계수의 의미: 절편 / 기울기 (독립변수 1증가시 평균적으로 종속변수가 기울기만큼 증가)
# F검정통계량 의미: 해당 모델이 평균 모델보다 좋은 성능을 보이는가?

# 계수별 t통계량 및 검정 , 신뢰구간
# H0: Bi 가 0이다. 0이 아니다.

# R square 값이 커지면? => 전체 데이터의 변동성 중 해당 모델이 설명하는 부분이 커진다.
# Adjusted R square : 독립변수의 개수가 늘어날수록 패널티를 준다.
# 선형모델의 성능을 평가하는 지표. 그저 이게 높다고해서 모델이 좋다는건 아님.
# 잔차를 시각화해봐야함. 잔차는 패턴이 있어서는 안 됨


import numpy as np

# 두 변수 X, Y
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 6, 8, 10])

# 공분산 계산
cov_matrix = np.cov(X, Y)
print("공분산 행렬:\n", cov_matrix)
print("공분산 값:", cov_matrix[0, 1])


# 상관계수 계산
corr_matrix = np.corrcoef(X, Y)
print("상관계수 행렬:\n", corr_matrix)


import matplotlib.pyplot as plt

plt.scatter(X, Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('X와 Y의 관계 시각화')
plt.grid(True)
plt.show()


import pandas as pd
from sklearn.datasets import load_iris

# 1. 아이리스 데이터 불러오기
df_iris = load_iris() 

# 2. pandas DataFrame으로 변환
iris = pd.DataFrame(data=df_iris.data, columns=df_iris.feature_names)
iris.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] #컬럼명 변경시

# 3. 타겟(클래스) 추가
iris["species"] = df_iris.target

# 4. 클래스 라벨을 실제 이름으로 변환 (0: setosa, 1: versicolor, 2: virginica)
iris["species"] = iris["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})





import seaborn as sns

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Iris 변수 간 상관계수 히트맵")
plt.show()


import statsmodels.formula.api as smf

model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Width", # 종속변수 ~ 독립변수1 + 독립변수2
                data=iris).fit()
print(model.summary())



model = smf.ols("Petal_Length ~ Petal_Width + Sepal_Width", data=iris).fit()


import numpy as np
import matplotlib.pyplot as plt

# 공분산 행렬 설정
mean = [3, 4]
cov = [[2**2, 3.6], [3.6, 3**2]]

# 데이터 생성
np.random.seed(42)
data = np.random.multivariate_normal(mean, cov, size=1000)
x1, x2 = data[:, 0], data[:, 1]

# 공분산, 상관계수 계산
cov_val = np.cov(x1, x2)[0, 1]
corr_val = np.corrcoef(x1, x2)[0, 1]

print(f"공분산: {cov_val:.2f}, 상관계수: {corr_val:.2f}")



plt.scatter(x1, x2)
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("X1과 X2의 관계")
plt.grid(True)
plt.show()


import numpy as np

# 독립변수: 상수항 포함
X = np.column_stack([np.ones(len(iris)), iris['Petal_Width'], iris['Sepal_Width']])
y = iris['Petal_Length'].values

# 정규방정식으로 회귀계수 계산
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print("벡터형 계수:", beta)

# 벡터형 계수: [ 2.25816351  2.15561052 -0.35503458]