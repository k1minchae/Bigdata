import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# 문제 1: 확률변수 X의 기댓값 구하기
x = np.array([1, 2, 3])
p_x = np.array([0.2, 0.5, 0.3])

e_x = np.sum(x * p_x)
print(f"1번: E(X) = {e_x:.3f}")



# 문제 2: 확률변수 X의 분산 구하기
var_x = np.sum((x - e_x)**2 * p_x)
print("2번: Var(X) =", var_x)



# 문제 3: Y = 2X + 3일때, 기댓값 E(Y)는?
# 기대값의 선형성 : 
# E(Y) = E(2X + 3) = 2E(X) + 3
e_y = 2 * e_x + 3
print("3번: E(Y) =", e_y)



# 문제 4: Var(Y) 구하기
# Var(Y) = Var(aX + 3) = a^2 * Var(X)
var_y = 2**2 * var_x
print("4번: Var(Y) =", var_y)




# 문제 5: 확률변수 X의 기댓값과 분산을 구하세요.
x = np.array([0, 1, 2, 3])
p_x = np.array([0.1, 0.3, 0.4, 0.2])

e_x = np.sum(x * p_x)
var_x = np.sum((x - e_x)**2 * p_x)
print(f"5번: 기댓값: {e_x:.2f}, 분산: {var_x}")




# 문제 6: 동전을 3번 던질 때 
# 앞면이 나오는 횟수를 확률변수 X라고 할 때
# Y = X1 + X2 + X3
# 기댓값 E(Y) 와 분산 Var(Y) 를 구하세요.
x = np.array([0, 1])
p_x = np.array([0.5, 0.5])
e_x = np.sum(x * p_x)
var_x = np.sum((x - e_x)**2 * p_x)

# 분산 구하는 또 다른 식
var_x = np.sum((x**2) * p_x) - e_x **2

e_y = e_x * 3
var_y = 3 ** 2 * var_x
print(f"6번: 기댓값: {e_y:.2f}, 분산: {var_y}")




# 문제 7: 공정한 6면체 주사위를 한 번 던졌을 때, 
# 나오는 눈의 수를 확률변수 X라고 할 때
# 기댓값과 분산은?
x = np.arange(1, 7)
p_x = np.repeat(1/6, repeats=6)
e_x = np.sum(x * p_x)
var_x = np.sum((x - e_x)**2 * p_x)

print(f"7번: 기댓값: {e_x:.2f}, 분산: {var_x:.2f}")




# 문제 8: 기댓값 E(X) = 5, E(Y) = 3 일때
# E(2X - Y + 4) 를 구하세요.
e_x = 5
e_y = 3
result = 2 * e_x - e_y + 4
print("8번: E(2X - Y + 4) =", result)




# 문제 9: E(X) = m, Var(X) = s^2 일때, 
# 확률변수 Z = aX + b의 기댓값과 분산은?

# E(Z) = a * m + b
# Var(Z) = a^2 * s^2




# 문제 10: 확률변수 X에서 p의 값을 구하고 기댓값 E(X)를 구하세요.
p = 1 - (0.3 + 0.4)
x = np.array([1, 2, 3])
p_x = np.array([0.3, p, 0.4])
e_x = np.sum(x * p_x)
print(f"10번: p: {p:.2f}, 기댓값: {e_x}")




# 문제 11: E(X), E(X^2), Var(X) 를 모두 구하세요.
x = np.array([1, 2, 4])
p_x = np.array([0.2, 0.5, 0.3])

e_x = np.sum(x * p_x)
e_x2 = np.sum((x**2) * p_x)
var_x = np.sum((x - e_x)**2 * p_x)
print(f"11번: E(X): {e_x:.2f}, E(X^2): {e_x2:.2f}, Var(X): {var_x:.2f}")

