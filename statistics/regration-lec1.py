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


# 1차원 경사 하강법
# K(n + 1) = k(n) + step * (- f'(k(n)))


# 2차원 경사 하강법
# [[B1], [B2]](n+1) = [[B1], [B2]](n) + (step(x) * - [[f'(B1)], [f'(B2)]])



'''
실습

'''
# 1. f(x) = 4(x - 2)^3 의 최소값을 경사하강법을 통해서 찾아보세요.
def my_f(x):
    return 4 * (x - 2)**3

def my_f_prime(x):
    return 12 * (x - 2)**2

k = np.linspace(-1, 5, 100)
plt.plot(k, my_f(k), color='blue', label='f(x) = 4(x - 2)^3')
plt.plot(k, my_f_prime(k), color='red', label="f'(x)")




# 2. 데이터 x1, x2, ..., x7은 다음 숫자와 같습니다.
x = np.array([4, 7, 13, 2, 1, 5, 9])

# 함수 f(B) = sum((xi - B)^2) 일 때 f(B)의 최소값을 만드는 B값을 경사하강법을 통해 찾아보세요.
import numpy as np
import matplotlib.pyplot as plt

# 데이터
x = np.array([4, 7, 13, 2, 1, 5, 9])

# 손실 함수
def f(beta):
    return np.sum((x - beta) ** 2)

# 기울기
def grad_f(beta):
    return -2 * np.sum(x - beta)

# 경사하강법 설정
epochs = 50
learning_rate = 0.01
B_vals = []
B = 0.0  # 초기값

# 수렴 과정 저장
for _ in range(epochs):
    B_vals.append(B)
    B -= learning_rate * grad_f(B)

B_vals = np.array(B_vals)
f_vals = np.array([f(b) for b in B_vals])

# 곡선 그리기
k = np.linspace(-1, 15, 200)
f_curve = np.array([f(beta) for beta in k])

# 애니메이션처럼 그리기
plt.figure(figsize=(10, 6))
plt.plot(k, f_curve, color='blue', label=r'$f(B) = \sum{(x_i - B)^2}$')
plt.axvline(np.mean(x), color='red', linestyle='--', label=f'평균 = {np.mean(x):.2f}')
plt.scatter(B_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')
plt.plot(B_vals, f_vals, color='orange', linestyle='--', alpha=0.5)

# 마지막 점 강조
plt.scatter(B_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')

plt.title('경사하강법 수렴 과정')
plt.xlabel('B')
plt.ylabel('f(B)')
plt.legend()
plt.grid(True)
plt.show()


# 3. f(x1, x2) = 4(x1^2) + 2(x2 - 1/2)^2
# 경사하강법을 통해서 f(x1, x2)의 최소값을 찾아보세요.


# 문제 4번
x = np.array([4, 7, 13, 2, 1, 5, 9])
y = np.array([1, 3, 5, 7, 2, 3, 2])
# f(B1, B2) = (y1 - B1X1 - B2)^2 + (y2 - B1X2 - B2)^2 + ... + (yn - B1Xn - B2)^2
# B1, B2를 경사하강법으로 구해보세요.