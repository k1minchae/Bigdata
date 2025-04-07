# 회귀 분석
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

def my_f(x):
    return x**2

sample = np.linspace(-10, 10, 100)
f_x = my_f(sample)



# 미분의 정의
# f'(x) = lim(h->0) (f(x + h) - f(x)) / h
# 입력값 : x=2, h=0.1
# ( f(2 + 0.1) - f(2) )  / 0.1

# f_x의 도함수
# (f(x + h) - f(x)) / h

x = 2
h = 0.000001
(my_f(x + h) - my_f(x)) / h

plt.plot(sample, f_x, color='blue', label='f(x) = x^2')
plt.axvline(x=-x, color='red', linestyle='--', label='x = -2')
plt.axvline(x=x, color='red', linestyle='--', label='x = 2')
plt.title('f(x) = x^2')
plt.xlabel('x')
plt.xticks(np.arange(-10, 10, 2))
plt.ylabel('f(x)')
plt.show()


def my_f_prime(x):
    h = 0.000001
    return (my_f(x + h) - my_f(x)) / h


my_f_prime(2)
my_f_prime(-2)

sample = np.linspace(-10, 10, 100)
plt.plot(sample, my_f_prime(sample), color='red')
plt.title("f'(x) = 2x")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.axvline(x=-5, color='black', linestyle='--', label='x = -2')
plt.axhline(y=-10, color='black', linestyle='--', label='y = 10')
plt.show()


def my_f(X):
    return np.sin(X)

def my_f(x):
    return x**3

############같이그려보자######################
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(sample, my_f(sample), color='blue', label='f(x) = x^2')
plt.title('f(x) = x^2')
plt.xlabel('x')
plt.xticks(np.arange(-10, 10, 2))
plt.ylabel('f(x)')

plt.subplot(1, 2, 2)
plt.plot(sample, my_f_prime(sample), color='red')
plt.title("f'(x) = 2x")
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.show()

# 도함수를 사용하면 특정 목적을 달성할 때 좋을 것 같다.
# 예를 들어, 특정 점에서의 기울기를 구할 수 있다.
# 도함수는 기울기를 나타내기 때문에, 기울기가 0이 되는 점은 최저가 되는 점이다.
# 도함수를 사용하면 특정 목적을 달성할 때 좋을 것 같다.

# 경사하강법
# 경사하강법은 기울기를 이용하여 최적의 값을 찾는 방법이다.
# 기울기가 0이 되는 점은 최저가 되는 점이다.


# f(B1, B2) = (B1 - 1)^2 + (B2 - 2)^2
# f(2, 3) = (2 - 1)^2 + (3 - 2)^2 = 1 + 1 = 2

# 입력값 2개인 함수
def my_f2(beta1, beta2):
    return (beta1-1) **2 + (beta2-2) **2

my_f2(2, 3)

# 3차원 평면에 해당 함수를 그려보세요.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 그래프 도구
# beta1과 beta2 값의 범위 설정
beta1 = np.linspace(-1, 3, 100)
beta2 = np.linspace(0, 5, 100)

# 그리드 생성
B1, B2 = np.meshgrid(beta1, beta2)

# Z 값 계산
Z = my_f2(B1, B2)

# 3D 그래프 생성
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D Surface plot
ax.plot_surface(B1, B2, Z, cmap='viridis', edgecolor='none')
ax.set_title('f(B1,B2) 3D 그래프')
ax.set_xlabel('B1')
ax.set_ylabel('B2')
ax.set_zlabel('f(B1, B2)')

plt.show()




# 등고선 그래프
plt.figure(figsize=(8, 6))
contour = plt.contourf(B1, B2, Z, levels=30, cmap='viridis')  # 채워진 등고선
contour_lines = plt.contour(B1, B2, Z, levels=30, colors='black', linewidths=0.5)  # 선 추가
plt.colorbar(contour)  # 색상 바 추가
plt.title('f(B1, B2) 등고선 그래프 (Top View)')
plt.xlabel('B1')
plt.ylabel('B2')
plt.axis('equal')
plt.show()




#### 한슬 코드 #### 
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 함수 및 기울기 정의
x, y = sp.symbols('x y')
f = (x - 1)**2 + (y - 2)**2
grad = [sp.diff(f, var) for var in (x, y)]
f_func = sp.lambdify((x, y), f, 'numpy')
grad_func = [sp.lambdify((x, y), g, 'numpy') for g in grad]

# 경사 하강법
alpha = 0.1
x0, y0 = -0.5, 3.5
path = [(x0, y0, f_func(x0, y0))]
for _ in range(30):
    dx, dy = grad_func[0](x0, y0), grad_func[1](x0, y0)
    x0 -= alpha * dx
    y0 -= alpha * dy
    path.append((x0, y0, f_func(x0, y0)))
xs, ys, zs = zip(*path)

# 표면 생성
x_vals = np.linspace(-1, 3, 200)
y_vals = np.linspace(0, 4, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_func(X, Y)

# Plotly 시각화
fig = go.Figure()

# 표면
fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8))

# 경로
fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='markers+lines',
                           marker=dict(size=4, color='red'),
                           line=dict(color='red', width=4),
                           name='Gradient Descent Path'))

# 최소점
fig.add_trace(go.Scatter3d(x=[1], y=[2], z=[0], mode='markers',
                           marker=dict(size=6, color='black'),
                           name='Minimum'))

fig.update_layout(title="Interactive Gradient Descent",
                  scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='f(x, y)'),
                  width=900, height=700)

fig.show()



import sympy as sp
import numpy as np
import plotly.graph_objects as go

# 함수 및 기울기 정의
x, y = sp.symbols('x y')
f = (x - 1)**2 + (y - 2)**2
grad = [sp.diff(f, var) for var in (x, y)]
f_func = sp.lambdify((x, y), f, 'numpy')
grad_func = [sp.lambdify((x, y), g, 'numpy') for g in grad]

# 경사 하강법 파라미터
alpha = 0.1
x0, y0 = -0.5, 3.5
path = [(x0, y0)]
for _ in range(30):
    dx, dy = grad_func[0](x0, y0), grad_func[1](x0, y0)
    x0 -= alpha * dx
    y0 -= alpha * dy
    path.append((x0, y0))
xs, ys = zip(*path)

# 등고선용 데이터
x_vals = np.linspace(-1, 3, 300)
y_vals = np.linspace(0, 4, 300)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_func(X, Y)

# 등고선 + 경로 Plotly 시각화
fig = go.Figure()

# 등고선
fig.add_trace(go.Contour(
    z=Z, x=x_vals, y=y_vals,
    contours_coloring='lines',
    line_width=2,
    colorscale='Viridis',
    showscale=True
))

# 경사 하강법 경로
fig.add_trace(go.Scatter(
    x=xs, y=ys,
    mode='markers+lines',
    marker=dict(color='red', size=6),
    line=dict(color='red', width=2),
    name='Gradient Descent Path'
))

# 최소값
fig.add_trace(go.Scatter(
    x=[1], y=[2],
    mode='markers',
    marker=dict(color='black', size=10),
    name='Minimum (1,2)'
))

# 레이아웃
fig.update_layout(
    title='2D Contour Plot with Gradient Descent Path',
    xaxis_title='x',
    yaxis_title='y',
    width=800,
    height=700
)

fig.show()

#############여기까지 한슬코드 ##############


# 1차원 경사 하강법
# K(n + 1) = k(n) + step * (- f'(k(n)))


# 2차원 경사 하강법
# [[B1], [B2]](n+1) = [[B1], [B2]](n) + (step(x) * - [[f'(B1)], [f'(B2)]])



'''
실습

'''
# 1 번
# 1. f(x) = 4(x - 2)^3 의 최소값을 경사하강법을 통해서 찾아보세요.

def my_f(x):
    return 4 * (x - 2)**3

def my_f_prime(x):
    return 12 * (x - 2)**2

k = np.linspace(-1, 5, 100)
plt.plot(k, my_f(k), color='blue', label=r'$f(x) = 4(x - 2)^3$')
plt.plot(k, my_f_prime(k), color='red', label=r"$f'(x) = 12(x - 2)^2$")
plt.xlabel('x')
plt.legend()
plt.show()

# fsolve 로 방정식 풀기
from scipy.optimize import fsolve
solution = fsolve(my_f_prime, x0=0)
print(f"f'(x) = 0 이 되는 x 값: {solution[0]:.2f}")


# 2 번
# 함수 f(B) = sum((xi - B)^2) 일 때 f(B)의 최소값을 만드는 B값을 경사하강법을 통해 찾아보세요.
x = np.array([4, 7, 13, 2, 1, 5, 9])


# 목적: 각 관측값 x_i와 추정값 B의 차이 제곱을 모두 더한 값 (최소화 목표)
def f(beta):
    return np.sum((x - beta) ** 2)

# 도함수 (기울기) 정의
# f(B)를 미분하면 f'(B) = -2 * sum(x - B)
# 이는 기울기이며, 경사하강법에서 다음 위치를 결정할 때 사용됨
def grad_f(beta):
    return -2 * np.sum(x - beta)

# 경사하강법(GD) 설정
epochs = 50            # 반복 횟수 (epoch 수)
learning_rate = 0.01   # 학습률 (한 번에 이동하는 정도)
B_vals = []            # 경로 저장용 리스트 (x축)
B = 0.0                # B의 초기값 (출발점)

# 경사하강법 반복 실행
for _ in range(epochs):
    B_vals.append(B)         # 현재 B값 저장
    B -= learning_rate * grad_f(B)  # 기울기 방향으로 B 업데이트

# 각 B에 대응하는 f(B) 값 계산 (y축)
B_vals = np.array(B_vals)
f_vals = np.array([f(b) for b in B_vals])

# 그래프를 위한 B 범위 설정
k = np.linspace(-1, 15, 200)       # B 축 범위
f_curve = np.array([f(beta) for beta in k])  # 곡선 전체를 위한 f(B)



# 시각화 시작
plt.figure(figsize=(10, 6))

# 손실 함수 곡선 그리기
plt.plot(k, f_curve, color='blue', label=r'$f(B) = \sum{(x_i - B)^2}$')

# 평균값(해석적 최솟값)에 수직선 표시
plt.axvline(np.mean(x), color='red', linestyle='--', label=f'최소값 = {np.mean(x):.2f}')

# 경사하강법이 이동한 경로를 점으로 표시
plt.scatter(B_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')

# 경사하강법 경로를 선으로 연결
plt.plot(B_vals, f_vals, color='orange', linestyle='--', alpha=0.5)

# 최종 수렴 지점 강조 (검은색 큰 점)
plt.scatter(B_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')

print('2번 답: ', round(B_vals[-1], 3))

# 제목 및 레이블
plt.title('경사하강법 수렴 과정')
plt.xlabel('B')
plt.ylabel('f(B)')
plt.legend()
plt.grid(True)
plt.show()







# 3. f(x1, x2) = 4(x1^2) + 2(x2 - 1/2)^2
# 경사하강법을 통해서 f(x1, x2)의 최소값을 찾아보세요.
# 함수 정의: f(x1, x2) = 4x1^2 + 2(x2 - 0.5)^2
# 함수 정의
def f(x1, x2):
    return 4 * x1**2 + 2 * (x2 - 0.5)**2

# 경사하강법으로 저장된 경로 (이전 코드에서 나온 path 사용)
# 경사하강법 수행
def grad_f(x1, x2):
    return np.array([8 * x1, 4 * (x2 - 0.5)])

x = np.array([3.0, 3.0])
learning_rate = 0.1     # 학습률 (한 번에 이동하는 정도)
epochs = 50000          # 반복 횟수 (epoch 수)
path = [x.copy()]       # 경로 저장용 리스트
for _ in range(epochs):
    # 현재 시점에서의 기울기 벡터 계산
    grad = grad_f(x[0], x[1])

    # 현재 위치 x에서, 기울기 방향으로 learning_rate만큼 반대 방향으로 이동
    # 최소값 찾기
    x -= learning_rate * grad
    path.append(x.copy())

# numpy array 로 변환
path = np.array(path)

# 그리드 생성
x1_vals = np.linspace(-3, 3, 100)
x2_vals = np.linspace(-1, 3, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f(X1, X2)

# 등고선 구간 설정
levels = np.linspace(np.min(Z), np.max(Z), 15)

# 등고선 선만 시각화
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=levels, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")  # 등고선 값 표시

# 경사하강법 경로 그리기
plt.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=4, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 그래프 설정
plt.title(r"$f(x_1, x_2) = 4x_1^2 + 2(x_2 - \frac{1}{2})^2$")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
print(f"최종 수렴값: B1 = {path[-1, 0]}, B2 = {path[-1, 1]}")




# 문제 4번
x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])
# f(B1, B2) = (y1 - B1X1 - B2)^2 + (y2 - B1X2 - B2)^2 + ... + (yn - B1Xn - B2)^2
# B1, B2를 경사하강법으로 구해보세요.

# 손실 함수 정의
def f(B1, B2):
    return np.sum((y - (B1 * x + B2)) ** 2)

# 미분한 함수
def grad_f(B1, B2):
    error = y - (B1 * x + B2)
    dB1 = -2 * np.sum(error * x)
    dB2 = -2 * np.sum(error)
    return np.array([dB1, dB2])

# 초기값, 학습률, 반복 수
B = np.array([0.0, 0.0])  # [B1, B2]
learning_rate = 0.001
epochs = 10_000_000
path = [B.copy()]

# 경사하강법 실행
for _ in range(epochs):
    grad = grad_f(B[0], B[1])
    B -= learning_rate * grad
    path.append(B.copy())

# numpy array로 변환
path = np.array(path)

# 등고선 그리기 위한 그리드 준비
B1_vals = np.linspace(-0.25, 0.75, 100)
B2_vals = np.linspace(-2, 6, 100)
B1_grid, B2_grid = np.meshgrid(B1_vals, B2_vals)

# 손실 함수 값 계산 (벡터화)
def full_loss_grid(B1_grid, B2_grid):
    total = np.zeros_like(B1_grid)
    for i in range(len(x)):
        total += (y[i] - (B1_grid * x[i] + B2_grid)) ** 2
    return total

Z = full_loss_grid(B1_grid, B2_grid)

# 등고선 시각화
plt.figure(figsize=(10, 6))
contours = plt.contour(B1_grid, B2_grid, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# 경사하강법 경로 표시
plt.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=4, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 시각화 설정
plt.title(r"$f(B_1, B_2) = \sum (y_i - B_1 x_i - B_2)^2$")
plt.xlabel("B1 (기울기)")
plt.ylabel("B2 (절편)")
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print(f"최종 수렴값: B1 = {path[-1, 0]:.4f}, B2 = {path[-1, 1]:.4f}")
