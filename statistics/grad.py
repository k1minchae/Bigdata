# 2조: 이주형, 김민채, 남원정, 송성필, 이주연

import matplotlib.pyplot as plt
import numpy as np

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
final_x, x_vals = gradient_descent(3, step=0.04, n=30)
final_x, x_vals

f_vals = f_x(x_vals)
f_vals

len(x_vals)
len(f_vals)
# 시각화
# x축 범위
k = np.linspace(0, 4, 300)
f_curve = f_x(k)
plt.figure(figsize=(10, 6))

# 함수 곡선
plt.plot(k, f_curve, color='blue', label=r'$f(x) = 4(x - 2)^2$')

# 경사하강법 경로
plt.scatter(x=x_vals, y=f_vals, color='orange', zorder=5, label='경사하강법 경로')
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

# 각 β에 대응하는 f(β) 값 계산
f_vals = np.array([f_beta(x, b) for b in B_vals])

# 시각화 시작
# 손실 함수 곡선 그리기 위한 범위 설정
k = np.linspace(-1, 15, 200)
f_curve = np.array([f_beta(x, beta) for beta in k])

plt.figure(figsize=(10, 6))
# 손실 함수 곡선
plt.plot(k, f_curve, color='blue', label=r'$f(β) = \sum{(x_i - β)^2}$')
# 해석적 최소값 (평균값) 표시
plt.axvline(np.mean(x), color='red', linestyle='--', label=f'최소값 = {np.mean(x):.2f}')
plt.scatter(B_vals, f_vals, color='orange', zorder=5, label='경사하강법 경로')


# 최종 수렴 지점 강조
plt.scatter(B_vals[-1], f_vals[-1], color='black', s=100, label='최종 수렴 지점')
plt.plot(B_vals, f_vals, color='orange', linestyle='--', alpha=0.5)

# 제목 및 축 라벨
plt.title('경사하강법 수렴 과정 (gradient_descent_2 버전)')
plt.xlabel('β')
plt.ylabel('f(β)')
plt.legend()
plt.grid(True)
plt.show()

# 결과 출력
print(f"최종 수렴값 β = {round(B_vals[-1], 3)}")




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
path = gradient_descent_3(init_beta1=0.2, init_beta2=1.0, step=0.1, n=1000)


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


# 결과 출력
print(f"최종 수렴값 x1 = {round(path[-1][0], 3)}, x2 = {round(path[-1][1], 3)}")



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

print(f"수렴한 β1: {beta_opt[0]}, 수렴한 β2: {beta_opt[1]}")
print(f"최소값 f(β): {np.sum((y - X.dot(beta_opt))**2)}")


#############
# lr = 0.0001
# n_iters = 50000000 
# 수렴한 β1: 0.04087193460496675, 수렴한 β2: 3.0463215258850362
# 최소값 f(β): 25.253405994550405

np.random.seed(408)
np.random.randint(1, 8)




# sympy 사용하기
import sympy as sp

# x를 정의
x = sp.symbols('x')

# 함수 f(x)정의
f = 4 * (x - 2)**2

# 도함수 정의
f_diff = sp.diff(f, x)

# sympy 함수들을 numpy 에서 사용할 수 있도록 변환
f_np = sp.lambdify(x, f, 'numpy')
f_diff_np = sp.lambdify(x, f_diff, 'numpy')



