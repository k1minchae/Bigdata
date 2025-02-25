# numpy 연습문제
import numpy as np

'''
연습 문제1

주어진 벡터의 각 요소에 5를 더한 새로운 벡터를 생성하세요
'''
a = np.array([1, 2, 3, 4, 5])
a += 5


'''
연습 문제2

주어진 벡터의 홀수 번째 요소만 추출하여 새로운 벡터를 생성하세요
'''
a = np.array([12, 21, 35, 48, 5])
a[a % 2 == 1]

'''
연습 문제3

주어진 벡터에서 최대값을 찾으세요
'''
a = np.array([1, 22, 93, 64, 54])
a[a == a.max()]

'''
연습 문제4

주어진 벡터에서 중복된 값을 제거한 새로운 벡터를 생성하세요
'''
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

'''
연습 문제5

주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요

결과: array([21, 24, 31, 44, 58, 67])
'''
a = np.array([21, 31, 58])
b = np.array([24, 44, 67])
c = np.zeros(a.size + b.size)
c[::2] = a
c[1::2] = b
c

'''
연습 문제6

다음 a 벡터의 마지막 값은 제외한 두 벡터 a와 b를 더한 결과를 구하세요

'''
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
a[:-1] + b



'''
연습 문제 7

주어진 벡터에서 가장 자주 등장하는 숫자를 찾아 출력하세요

'''
a = np.array([1, 3, 3, 2, 1, 3, 4, 2, 2, 2, 5, 6, 6, 6, 6])
# cnt, val = np.unique(a, return_counts=True)
counts = np.unique_counts(a).counts
values = np.unique_counts(a).values
most = counts.max() == counts
values[most]



'''
연습 문제8

주어진 벡터에서 3의 배수만 추출한 새로운 벡터를 만드세요

'''
a = np.array([12, 5, 18, 21, 7, 9, 30, 25, 3, 6])
a[a % 3 == 0]


'''
연습 문제9

주어진 벡터를 중앙값을 기준으로 두 개의 벡터로 분리하세요

'''
a = np.array([10, 20, 5, 7, 15, 30, 25, 8])
a[np.median(a) >= a]
a[np.median(a) < a]


'''
연습 문제10

주어진 벡터에서 중앙값과 가장 가까운 값을 찾으시오

'''
a = np.array([12, 45, 8, 20, 33, 50, 19])
a_idx = np.where(np.median(a) == a)[0]
diff = np.abs(a - np.median(a))   # 중앙값과의 차이
diff[a_idx] = np.nan
a[np.nanargmin(diff)]
print(f"중앙값: {np.median(a)}")
print(f"가장 가까운 값: {a[np.nanargmin(diff)]}")


'''
연습 문제11

다음과 같은 행렬 A를 만들어 보세요

## 행렬 A:
## [[3 5 7]
## [2 3 6]]
'''

a = np.array([[3, 5, 7], [2, 3, 6]])


'''
연습 문제12
다음과 같이 행렬 B가 주어졌을 때
2번째 4번째 5번째 행 만을 선택하여 3 by 4 행렬을 만들어보세요

## 행렬 B:
## [[ 8 10 7 8]
## [ 2 4 5 5]
## [ 7 6 1 7]
## [ 2 6 8 6]
## [ 9 3 4 2]]
'''
b = np.array([[8, 10, 7, 8], [2, 4, 5, 5], [7, 6, 1, 7], [2, 6, 8, 6], [9, 3, 4, 2]])
b[[1, 3, 4], :]


'''
연습 문제 13

연습 문제2에서 주어진 행렬 B에서 3번째 열의 값이 3보다 큰 행들만 골라내 보세요

'''
b[b[:, 2] > 3, :]


'''
연습 문제 14
주어진 행렬B의 행별로 합계를 내고 싶을 때 rowSums() 함수를 사용

각 행 별 합이 20보다 크거나 같은 행 만을 걸러내어 새로운 행렬을 작성해보세요

'''
row_sums = np.sum(b, axis=1)
b[row_sums >= 20, :]


'''
연습 문제 15
원래 주어진 행렬 B에서 
각 열별 평균이 5보다 크거나 같은 열이 몇 번째 열에 위치하는지
np.mean() 함수를 사용하여 알아내는 코드를 작성해보세요

'''

col_means = np.mean(b, axis=0)
np.where(col_means >= 5)[0]


'''
연습 문제 16

행렬B의 각 행에 7보다 큰 숫자가 하나라도 들어있는 행을 
걸러내는 코드를 작성해 주세요

'''
b[np.any(b > 7, axis=1), :]




'''
연습 문제 17

단순 선형 회귀 모델에 회귀 계수 (기울기) 구하는 식을 작성해 주세요

'''

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 기울기 구하기
np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)

'''
2025.02.25 (화)
'''


# 1. 두 행렬의 곱 구하기
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a @ b


# 2. 다음 두 행렬을 곱할 수 있는지 확인하고, 가능하면 계산하라.
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[7, 8], [9, 10], [11, 12]])
a.shape, b.shape    # (2, 3), (3, 2) => 가능
np.matmul(a, b)


# 3. 임의의 정사각 행렬 A와 단위 행렬 I가 주어졌을 때, AI와 IA를 구하고, 원래 행렬과 같은지 확인하라
a = np.array([[1, 2], [3, 4]])
i = np.array([[1, 0], [0, 1]])
ai = a @ i
ia = i @ a
np.array_equal(ai, a)   # True
np.array_equal(ia, a)   # True


# 4. 임의의 행렬 A와 영행렬 Z를 곱하면 어떤 결과가 나오는지 확인하라.
a = np.array([[1, 2], [3, 4]])
z = np.array([[0, 0], [0, 0]])
a.dot(z)
# array([[0, 0],
#        [0, 0]])


# 5. 대각 행렬 D와 행렬 A의 곱을 계산하라.
d = np.array([[2, 0], [0, 3]])
a = np.array([[4, 5], [6, 7]])
d.dot(a)
# 대각에만 원소가 있는 행렬: 대각행렬


# 6. 행렬 A와 벡터 v를 곱하여 결과를 구하라.
a = np.array([[1, 2], [3, 4], [5, 6]])
v = np.array([[0.4], [0.6]])
a.dot(v)    # A행들의 가중평균 값


# 7. 3D 행렬(텐서)과 행렬을 곱할수있는지 확인하고 가능하면 계산하라.
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])
t = np.array([a, b])

np.matmul(t, c)
# array([[[ 31,  34],
#         [ 71,  78]],   => np.matmul(a, c)

#        [[111, 122],
#         [151, 166]]])  => np.matmul(b, c)



# 8. 대칭 행렬 S가 주어졌을 때 SS의 결과를 확인하세요. S의 역행렬은 어떤 성질을 가지는지 확인하세요.
s = np.array([[2, -1], [-1, 2]])
np.linalg.det(s)
np.linalg.inv(s)        # 역행렬도 대칭이다.
ss = np.matmul(s, s)    # 결과 값도 대칭이다.


# 9. 세 개의 행렬을 차례로 곱하는 연산을 수행하라. (AB)C와 A(BC)의 결과를 비교해보세요!
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = np.array([[9, 10], [11, 12]])

vec1 = np.matmul(np.matmul(a, b), c)
vec2 = np.matmul(a, np.matmul(b, c))
np.array_equal(vec1, vec2)  # True
# 결과 값이 같으므로 계산이 편한 것 먼저 계산해도 된다.


# 10. 역행렬과 연립방정식
a = np.array([3, 2, -1, 2, -2, 4, -1, 0.5, -1]).reshape(3, 3)
np.linalg.det(a)
inv_a = np.linalg.inv(a)

b = np.array([1, -2, 0])
x = np.matmul(inv_a, b) # 해


# numpy 배열에 apply 함수 적용
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
array_2d.max(axis=0)
np.apply_along_axis(max, axis=0, arr=array_2d)  
np.apply_along_axis(np.mean, axis=1, arr=array_2d)     # 내가 만든 함수도 여기에 적용 가능

def my_sum(input):
    return sum(input + 1)

np.apply_along_axis(my_sum, axis=1, arr=array_2d)


'''
주어진 방정식을 활용하여 회귀계수를 구하세요.
'''
x = np.array([[2, 4, 6], [1, 7, 2], [7, 8, 12]])
y = np.array([[10], [5], [15]])
xt = x.transpose()  # x.T로 써도 됨
np.linalg.inv(x.T @ x) @ x.T @ y

# 라이브러리 활용해서 구한 값
# import statmodels.api as sm
# model = sm.OLS(y, x).fit()
# print("회귀계수 : ", model.params)