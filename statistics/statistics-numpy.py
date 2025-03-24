import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
베르누이 분포는 random variable의 1회 실행결과가 
1과 0의 2가지 이진값으로만 이루어진 분포를 의미합니다.

예시) 동전 던지기

'''
# 함수 이름: bernoulli()
# 입력값 X, 0, 1 둘 중 하나를 가짐
# rng.uniform(0, 1) 값이 0.5보다 작거나 같으면 1, 그렇지 않으면 0
def bernoulli():
    rng = np.random.default_rng()
    return int(rng.uniform(0, 1) <= 0.5)

cnt = 0
for i in range(10_000_000):
    cnt += bernoulli()
print(cnt / 10_000_000)   # 0.5001187



'''

확률 변수 이해하기

'''
# 확률변수  X는 0, 1, 2
# 대응하는 확률이 p와 같다. 
# 확률변수 X를 만들어보세요.
p = [0.2, 0.5, 0.3]

def event_x():
    rng = np.random.default_rng()
    u = rng.uniform(0, 1)
    if u <= 0.2:    # 0% ~ 20$
        return 0
    elif u <= 0.7:  # 20% ~ 70%
        return 1
    else:
        return 2
event_x()



import time
from itertools import product
# X(size = 10) 입력값을 만들어서 
# 결과값을 넘파이 벡터로 나올 수 있도록 개조

# numpy 를 활용한 연산
choice = np.array([0, 1, 2])
def X(size):
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size)
    condition = np.array([u <= 0.2, u <= 0.7, u < 1])
    return np.select(condition, choice)

# python 의 for loop 를 활용한 연산    
def for_X(size):
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size)
    for i in range(size):
        if u[i] <= 0.2:
            u[i] = 0
        elif u[i] <= 0.7:
            u[i] = 1
        else:
            u[i] = 2
    return u

# numpy 연산속도
start = time.time()
X(10_000_000)
end = time.time()
numpy_speed = end - start

# for loop 연산 속도
start = time.time()
for_X(10_000_000)
end = time.time()
for_loop_speed = end - start

# 연산 속도 차이 비교
print(f'numpy: {round(numpy_speed, 2)}, for_loop: {round(for_loop_speed, 2)}')
# numpy: 0.18, for_loop: 2.1


# 확률 변수 X의 확률 질량함수 시각화
x = np.array([0, 1, 2])
p_x = np.array([0.2, 0.5, 0.3])

plt.bar(x, p_x)
plt.xlabel('Random Variable X')
plt.ylabel('Probability')
plt.title('Probability of X')
plt.xticks(x)
plt.show()

# 확률변수의 기대값
μ = np.sum(x * p_x) # 모평균 E(X)

# 데이터 생성 및 시각화
data = X(200)
x̄ = data.mean() # 표본평균
# 표본 평균을 통해 모평균 추정 => 
sns.kdeplot(data, bw_method=0.5, fill=True)
plt.axvline(μ, color='red', linestyle='--', label='Expected Value')
plt.legend()
plt.show()


'''
평균이 같고 분산이 다른 데이터 알아보기

'''

y = np.array([-1, 1, 3])
p_y = np.array([0.2, 0.6, 0.2])
e_Y = np.sum(y * p_y)
var_Y = np.sum(p_y * (y - e_Y) ** 2)  # 분산 계산
std_Y = np.sqrt(var_Y)  # 표준편차 계산

x = np.array([0, 1, 2])
p_x = np.array([0.2, 0.6, 0.2])
e_X = np.sum(x * p_x)
var_X = np.sum(p_x * (x - e_X) ** 2)  # 분산 계산
std_X = np.sqrt(var_X)  # 표준편차 계산

plt.subplot(121)
plt.bar(x, p_x)
plt.title(f'var of X: {var_X:.2f}, std: {std_X:.2f}')
plt.subplot(122)
plt.bar(y, p_y)
plt.title(f'var of Y: {var_Y:.2f}, std: {std_Y:.2f}')
plt.show()



'''
문제: 확률변수 X 의 확률 분포표는 다음과 같습니다.
X=x: 1, 2, 3, 4
p(X=x): 0.1, 0.3, 0.2, 0.4

'''
x = np.array([1, 2, 3, 4])
p_x = np.array([0.1, 0.3, 0.2, 0.4])

# 1) 평균을 구하세요.
e_x = np.sum(x * p_x)


# 2) 분산을 구하세요.
var_x = np.sum(((x - e_x)**2) * p_x)


# 3) X에서 평균보다 큰 값이 나올 확률은 얼마인가요?
p_bigger = np.sum(p_x[x > e_x])


# 4) X의 확률 질량 함수를 Bar 그래프로 그려보세요.
plt.bar(x, p_x)
plt.xlabel('Random Variable X')
plt.ylabel('Probability')
plt.title('Probability of X')
plt.xticks(x)
plt.show()

# 5) 5개의 표본을 무작위로 추출한 값의 평균을 계산해보세요.
def X(size):
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size)
    condition = np.array([u <= 0.1, u <= 0.4, u <= 0.6, u < 1])
    return np.select(condition, x)
X(5).mean()

# 6) 4번에서 그린 그래프에, 확률 변수의 평균값을 빨간 세로 선으로 표시하고,
# 5번에서 계산한 표본 평균을 파란 세로선으로 표시하세요. (코드를 돌릴 때마다 값이 바뀜)
plt.bar(x, p_x)
plt.xlabel('Random Variable X')
plt.ylabel('Probability')
plt.xticks(x)
plt.title('Probability of X')
plt.axvline(e_x, label="μ", color="red")
plt.axvline(X(5).mean(), label="5 size mean", color="blue")
plt.axvline(X(20).mean(), label="20 size mean", color="green")
plt.legend()
plt.show()



# 7) 5개의 표본으로 계산한 표본 평균 300개 발생 -> 히스토그램
means_5_size = [X(5).mean() for _ in range(300)]
plt.figure(figsize=(20, 8))
plt.subplot(111)
plt.hist(means_5_size, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.xticks(x)
plt.title('sample : 5')

# 8) 20개의 표본으로 계산한 표본 평균 300개 발생 -> 히스토그램
means_20_size = [X(20).mean() for _ in range(300)]
plt.subplot(121)
plt.hist(means_20_size, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Sample Mean')
plt.ylabel('Frequency')
plt.title('sample : 20')
plt.show()


'''
문제 2
X1: 0과 1을 가지는 확률변수 (1이 나올 확률: 0.3)
X2: 0과 1을 가지는 확률변수 (1이 나올 확률: 0.3)
Y = X1 + X2 일 때 , Y의 확률분포표를 작성하세요.
단, X1, X2 는 서로 영향이 없음
'''
x = np.array([1, 0])
p_x = np.array([0.3, 0.7])
def X(size):
    rng = np.random.default_rng()
    u = rng.uniform(0, 1, size)
    condition = np.array([u <= 0.3, u < 1])
    return np.select(condition, x)
e_x = np.sum(x * p_x)
var_x = np.sum((x - e_x) ** 2 * p_x)

def Y(size):
    return X(size) + X(size)

# P(A & B) = p(A) * P(B|A)
# A: X1 = 0
# B: X2 = 0
# A와 B는 독립이므로 P(A) * P(B)
# 0.7 * 0.7


# P(Y=1) 을 구하려면?
# P(X1=0 & X2=1)
# P(X1=1 & X2=0)


# E(Y) 는?
e_y = e_x + e_x

# Var(Y) 는?
var_y = var_x + var_x


# Y = X1 + X2 + X3 + X4 + X5 + X6 + X7 일 때 각각의 확률 계산

# X1, X2, ..., X7의 가능한 값 (0 또는 1)
x_values = [0, 1]
p_x = [0.7, 0.3]  # 확률: 0이 나올 확률 0.7, 1이 나올 확률 0.3

# 모든 가능한 조합 생성
combinations = list(product(x_values, repeat=4))

# Y 값과 그에 해당하는 확률 계산
y_probabilities = np.zeros(5)  # Y는 0부터 7까지의 값을 가질 수 있음
for combination in combinations:
    y_value = sum(combination)
    prob = np.prod([p_x[val] for val in combination])
    y_probabilities[y_value] += prob

# 결과 출력
for y_value, prob in enumerate(y_probabilities):
    print(f"P(Y={y_value}) = {prob:.4f}")