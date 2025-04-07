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
    return 4 * (x - 2)**2

def my_f_prime(x):
    return 8 * (x - 2)

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






def f_x(x):
    return 4*((x-2)**2)

def f_p_x(x):
    return 8 * (x - 2)

def gradient_descent(x, step, n):
    history = np.zeros(n+1)
    history[0]=x
    for i in range(1, n+1):
        x = x - step*f_p_x(x)
        history[i] = x
    return x, history


gradient_descent(3, 0.02, 50)

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



# 시각화
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









# 2. 주형님 코드
x = np.array([4,7,13,2,1,5,9])

def f_beta(x, beta):
    return sum((x-beta)**2)

def f_prime_beta(x, beta):
    return -2 * sum(x-beta)


def gradient_descent_2(x, init_beta, step, n):
    beta = init_beta
    history = np.zeros(n + 1)
    history[0] = beta
    for i in range(1, n + 1):
        beta = beta - step * f_prime_beta(x, beta)
        history[i] = beta
    return beta, history

# 실행
final_beta, B_vals = gradient_descent_2(x, init_beta=3.0, step=0.01, n=50)

# 각 B에 대응하는 f(B) 값 계산
f_vals = np.array([f_beta(x, b) for b in B_vals])

# 손실 함수 곡선 그리기 위한 범위 설정
k = np.linspace(-1, 15, 200)
f_curve = np.array([f_beta(x, beta) for beta in k])

# 시각화 시작
plt.figure(figsize=(10, 6))

# 손실 함수 곡선
plt.plot(k, f_curve, color='blue', label=r'$f(B) = \sum{(x_i - B)^2}$')

# 해석적 최소값 (평균값) 표시
plt.axvline(np.mean(x), color='red', linestyle='--', label=f'최소값 = {np.mean(x):.2f}')

plt.scatter(B_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')



# 최종 수렴 지점 강조
plt.scatter(B_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')


plt.plot(B_vals, f_vals, color='orange', linestyle='--', alpha=0.5)

# 제목 및 축 라벨
plt.title('경사하강법 수렴 과정 (gradient_descent_2 버전)')
plt.xlabel('B')
plt.ylabel('f(B)')
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print(f"최종 수렴값 B = {round(B_vals[-1], 3)}")


f_beta(x, 1)
f_prime_beta(x, 1)







# 3. f(x1, x2) = 4(x1^2) + 2(x2 - 1/2)^2
# 경사하강법을 통해서 f(x1, x2)의 최소값을 찾아보세요.

# 함수 정의
def f(x1, x2):
    return 4 * x1**2 + 2 * (x2 - 0.5)**2

# 미분
def grad_f(x1, x2):
    return np.array([8 * x1, 4 * (x2 - 0.5)])

# 경사하강법 수행
x = np.array([3.0, 3.0])
learning_rate = 0.01     # 학습률 (한 번에 이동하는 정도)
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
plt.plot(path[:, 0], path[:, 1], coLor='red', marker='o', markersize=4, label='경사하강법 경로')
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
# f(B1, B2) = (y1 - B1X1 - B2)^2 + (y2 - B1X2 - B2)^2 + ... + (yn - B1Xn - B2)^2
# B1, B2를 경사하강법으로 구해보세요      
# 1. 학습 데이터 (x, y)
x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])

# 2. 손실 함수 정의
# 목적: B1, B2가 주어졌을 때 오차 제곱합(예측값과 실제값의 차이)을 계산
def f(B1, B2):
    y_pred = B1 * x + B2  # 예측값
    return np.sum((y - y_pred) ** 2)

# 3. 기울기 함수 정의 (도함수)
# B1과 B2에 대한 편미분값을 리턴 → 경사하강법에서 사용할 방향
def grad_f(B1, B2):
    error = y - (B1 * x + B2)      # 예측 오차 벡터
    dB1 = -2 * np.sum(error * x)   # B1에 대한 편미분
    dB2 = -2 * np.sum(error)       # B2에 대한 편미분
    return np.array([dB1, dB2])    # 기울기 벡터 반환

# 4. 경사하강법 초기 설정
B = np.array([0.0, 0.0])      # B = [B1, B2] 초기값 (0, 0)에서 시작
learning_rate = 0.001         # 러닝레이트 (한 번에 이동하는 크기)
epochs = 10000               # 반복 횟수
path = [B.copy()]             # 경로 기록을 위한 리스트

# 5. 경사하강법 반복 실행
for _ in range(epochs):
    grad = grad_f(B[0], B[1])     # 현재 위치에서 기울기 계산
    B -= learning_rate * grad     # 기울기 반대 방향으로 이동 (최소값 방향)
    path.append(B.copy())         # 이동한 위치 저장 (시각화용)

# 6. 배열로 변환 (path는 좌표 기록)
path = np.array(path)

# 7. 등고선 시각화를 위한 손실값 계산
# B1, B2 값들의 조합으로 그리드 생성
B1_vals = np.linspace(-0.25, 0.75, 100)
B2_vals = np.linspace(-2, 6, 100)
B1_grid, B2_grid = np.meshgrid(B1_vals, B2_vals)

# 8. 각 (B1, B2) 조합마다 손실값 계산해서 등고선용 Z 만들기
def full_loss_grid(B1_grid, B2_grid):
    total = np.zeros_like(B1_grid)
    for i in range(len(x)):
        total += (y[i] - (B1_grid * x[i] + B2_grid)) ** 2  # 벡터화 연산
    return total

Z = full_loss_grid(B1_grid, B2_grid)

# 9. 등고선 시각화 시작
plt.figure(figsize=(10, 6))
contours = plt.contour(B1_grid, B2_grid, Z, levels=30, cmap='viridis')  # 손실 함수의 등고선
plt.clabel(contours, inline=True, fontsize=8)                            # 등고선 수치 라벨 표시

# 10. 경사하강법 경로 시각화
plt.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=4, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 11. 시각화 마무리
plt.title(r"$f(B_1, B_2) = \sum (y_i - B_1 x_i - B_2)^2$")
plt.xlabel("B1 (기울기)")
plt.ylabel("B2 (절편)")
plt.legend()
plt.grid(True)
plt.show()

# 12. 최종 결과 출력
print(f"최종 수렴값: B1 = {path[-1, 0]:.4f}, B2 = {path[-1, 1]:.4f}")







# 1.


# 2.
x = np.array([4,7,13,2,1,5,9])

def f_beta(x, beta):
    return sum((x-beta)**2)

def f_prime_beta(x, beta):
    return -2 * sum(x-beta)

def gradient_descent_2(x, init_beta, step, n):
    beta = init_beta
    history = np.zeros(n+1)
    history[0]=beta
    for i in range(1, n+1):
        beta = beta - step*f_prime_beta(x, beta)
        history[i] = beta
    return beta, history


gradient_descent_2(x,3,0.1,50)


f_beta(x, 1)
f_prime_beta(x, 1)



# 3.

def f_x(x1, x2):
    return (4*(x1**2)) + 2*((x2-0.5)**2)

# f_x(0.5,0.5)

def f_p_x(beta1, beta2):
    beta1 = beta1*8
    beta2 = 4*(beta2-0.5)
    return beta1, beta2

def f_p_x1(beta1):
    return beta1*8

def f_p_x2(beta2):
    return 4*(beta2-0.5)


def gradient_descent_3(init_beta1, init_beta2, step, n):
    beta1 = init_beta1
    beta2 = init_beta2
    history = np.zeros(n*2).reshape(n,2)
    history[0][0]=beta1
    history[0][1]=beta2
    for i in range(1, n):
        beta1 = beta1 - step*f_p_x1(beta1)
        beta2 = beta2 - step*f_p_x2(beta2)
        history[i][0] = beta1
        history[i][1] = beta2
    return history


gradient_descent_3(0.0000001, 1, 0.1, 1000000)



# 4.
import numpy as np

# 예제 데이터

x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])
# 디자인 행렬 (n×2): [x_i, 1]
X = np.vstack([x, np.ones_like(x)]).T      # shape (n,2)

# 목적함수 f(β) = ||y - Xβ||^2
# 기울기 ∇f = -2 Xᵀ (y - Xβ)
def grad(beta):
    # beta: shape (2,)
    return -2 * X.T.dot(y - X.dot(beta))

# 경사하강법
def gradient_descent_linreg(X, y, init_beta, lr=0.01, n_iters=1000):
    beta = init_beta.astype(float)
    history = np.zeros((n_iters+1, 2))
    history[0] = beta
    for i in range(1, n_iters+1):
        beta -= lr * grad(beta)
        history[i] = beta
    return beta, history

# 초기값, 학습률, 반복횟수 설정
init_beta = np.array([0.0, 0.0])  # [β1, β2]
lr = 0.0001
n_iters = 500000

# 실행
beta_opt, path = gradient_descent_linreg(X, y, init_beta, lr, n_iters)
print("수렴한 β:", beta_opt)
print("최소값 f(β):", np.sum((y - X.dot(beta_opt))**2))


import matplotlib.pyplot as plt

# 1. 그리드 생성 (β1, β2 범위)
b1_vals = np.linspace(-0.2, 0.8, 200)
b2_vals = np.linspace(-2, 6, 200)
B1, B2 = np.meshgrid(b1_vals, b2_vals)

# 2. 각 점에서의 손실함수 값 계산
Z = np.zeros_like(B1)
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        beta_tmp = np.array([B1[i, j], B2[i, j]])
        Z[i, j] = np.sum((y - X.dot(beta_tmp)) ** 2)

# 3. 등고선 시각화
plt.figure(figsize=(10, 6))
contours = plt.contour(B1, B2, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# 4. 경사하강법 경로 시각화
plt.plot(path[:, 0], path[:, 1], color='red', linestyle='--', marker='o', markersize=3, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 5. 시각화 설정
plt.title(r"$f(\beta_1, \beta_2) = \| y - X\beta \|^2$")
plt.xlabel(r"$\beta_1$ (기울기)")
plt.ylabel(r"$\beta_2$ (절편)")
plt.legend()
plt.grid(True)
plt.show()











# 2조: 이주형, 김민채, 남원정, 송성필, 이주연

# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 1번
def f_x(x):
    return 4*((x-2)**2)

def f_p_x(x):
    return 8 * (x - 2)

def gradient_descent(x, step, n):
    history = np.zeros(n+1)
    history[0]=x
    for i in range(1, n+1):
        x = x - step*f_p_x(x)
        history[i] = x
    return x, history


# 실행
final_x, x_vals = gradient_descent(3, step=0.04, n=20)
f_vals = f_x(x_vals)

# 시각화
# x축 범위
k = np.linspace(0, 4, 300)
f_curve = f_x(k)
plt.figure(figsize=(10, 6))

# 함수 곡선
plt.plot(k, f_curve, color='blue', label=r'$f(x) = 4(x - 2)^2$')

# 경사하강법 경로
plt.scatter(x_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')
plt.plot(x_vals, f_vals, color='orange', linestyle='--', alpha=0.6)

# 최솟값 x=2 표시
plt.axvline(x=2, color='red', linestyle='--', label='최소값 x=2')

# 최종 도착점
plt.scatter(x_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')

# 세부 설정
plt.title('경사하강법 수렴 시각화')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()

# 결과 확인
print(f"최종 수렴값 x = {round(x_vals[-1], 4)}")




# 2번
x = np.array([4,7,13,2,1,5,9])

def f_beta(x, beta):
    return sum((x-beta)**2)

def f_prime_beta(x, beta):
    return -2 * sum(x-beta)

def gradient_descent_2(x, init_beta, step, n):
    beta = init_beta
    history = np.zeros(n+1)
    history[0]=beta
    for i in range(1, n+1):
        beta = beta - step*f_prime_beta(x, beta)
        history[i] = beta
    return beta, history


# 실행
final_beta, B_vals = gradient_descent_2(x, init_beta=3.0, step=0.03, n=20)

# 각 B에 대응하는 f(B) 값 계산
f_vals = np.array([f_beta(x, b) for b in B_vals])

# 시각화 시작
# 손실 함수 곡선 그리기 위한 범위 설정
k = np.linspace(-1, 15, 200)
f_curve = np.array([f_beta(x, beta) for beta in k])

plt.figure(figsize=(10, 6))
# 손실 함수 곡선
plt.plot(k, f_curve, color='blue', label=r'$f(B) = \sum{(x_i - B)^2}$')
# 해석적 최소값 (평균값) 표시
plt.axvline(np.mean(x), color='red', linestyle='--', label=f'최소값 = {np.mean(x):.2f}')
plt.scatter(B_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')


# 최종 수렴 지점 강조
plt.scatter(B_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')
plt.plot(B_vals, f_vals, color='orange', linestyle='--', alpha=0.5)

# 제목 및 축 라벨
plt.title('경사하강법 수렴 과정 (gradient_descent_2 버전)')
plt.xlabel('B')
plt.ylabel('f(B)')
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print(f"최종 수렴값 B = {round(B_vals[-1], 3)}")




# 3번
def f_x(x1, x2):
    return (4*(x1**2)) + 2*((x2-0.5)**2)

# f_x(0.5,0.5)

def f_p_x(beta1, beta2):
    beta1 = beta1*8
    beta2 = 4*(beta2-0.5)
    return beta1, beta2

def f_p_x1(beta1):
    return beta1*8

def f_p_x2(beta2):
    return 4*(beta2-0.5)

def gradient_descent_3(init_beta1, init_beta2, step, n):
    beta1 = init_beta1
    beta2 = init_beta2
    history = np.zeros(n*2).reshape(n,2)
    history[0][0]=beta1
    history[0][1]=beta2
    for i in range(1, n):
        beta1 = beta1 - step*f_p_x1(beta1)
        beta2 = beta2 - step*f_p_x2(beta2)
        history[i][0] = beta1
        history[i][1] = beta2
    return history


# 실행
path = gradient_descent_3(init_beta1=0.0000001, init_beta2=1.0, step=0.1, n=100)


# 시각화
# 등고선용 그리드
x1_vals = np.linspace(-0.25, 0.25, 200)
x2_vals = np.linspace(0.4, 1.1, 200)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f_x(X1, X2)

# 등고선 시각화
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# 경사하강법 경로 시각화
plt.plot(path[:, 0], path[:, 1], color='red', linestyle='--', marker='o', markersize=3, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 그래프 설정
plt.title(r"$f(x_1, x_2) = 4x_1^2 + 2(x_2 - \frac{1}{2})^2$")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()



# 4번

x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])
# 디자인 행렬 (n×2): [x_i, 1]
X = np.vstack([x, np.ones_like(x)]).T      # shape (n,2)


def grad(beta):
    # beta: shape (2,)
    return -2 * X.T.dot(y - X.dot(beta))

# 경사하강법
def gradient_descent_linreg(X, y, init_beta, lr=0.01, n_iters=1000):
    beta = init_beta.astype(float)
    history = np.zeros((n_iters+1, 2))
    history[0] = beta
    for i in range(1, n_iters+1):
        beta -= lr * grad(beta)
        history[i] = beta
    return beta, history

# 초기값, 학습률, 반복횟수 설정
init_beta = np.array([0.0, 0.0])  # [β1, β2]
lr = 0.0001
n_iters = 500000


# 실행
beta_opt, path = gradient_descent_linreg(X, y, init_beta, lr, n_iters)
print("수렴한 β:", beta_opt)
print("최소값 f(β):", np.sum((y - X.dot(beta_opt))**2))


# 시각화
# 1. 그리드 생성 (β1, β2 범위)
b1_vals = np.linspace(-0.2, 0.8, 200)
b2_vals = np.linspace(-2, 6, 200)
B1, B2 = np.meshgrid(b1_vals, b2_vals)

# 2. 각 점에서의 손실함수 값 계산
Z = np.zeros_like(B1)
for i in range(B1.shape[0]):
    for j in range(B1.shape[1]):
        beta_tmp = np.array([B1[i, j], B2[i, j]])
        Z[i, j] = np.sum((y - X.dot(beta_tmp)) ** 2)

# 3. 등고선 시각화
plt.figure(figsize=(10, 6))
contours = plt.contour(B1, B2, Z, levels=30, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# 4. 경사하강법 경로 시각화
plt.plot(path[:, 0], path[:, 1], color='red', linestyle='--', marker='o', markersize=3, label='경사하강법 경로')
plt.scatter(path[-1, 0], path[-1, 1], color='black', s=100, label='최종 수렴점')

# 5. 시각화 설정
plt.title(r"$f(\beta_1, \beta_2) = \| y - X\beta \|^2$")
plt.xlabel(r"$\beta_1$ (기울기)")
plt.ylabel(r"$\beta_2$ (절편)")
plt.legend()
plt.grid(True)
plt.show()
