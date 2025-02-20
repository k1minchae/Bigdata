import numpy as np

'''
[벡터 생성하기 예제]
numpy 의 벡터는 같은 자료형으로 채워진다
'''
a = np.array([1, 2, 3, 4, 5]) # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])
c = np.array([True, False, True, True])
d = np.array([1, 2, 3, 4, "5"])
print("숫자형 벡터: ", a)

# 빈 배열 선언 후 채우기
x = np.empty(3) # np.zeros()
x[0] = 3
x[1] = 3
x[2] = 3

# 파이썬의 range보다 유연함
np.arange(1, 11) # 매개변수: start, stop, step, dtype=float64
np.arange(1, 11, 0.5)

# 시작점 종료점 사이의 균일한 간격의 배열 생성
np.linspace(1, 3, 3) # 끝값 포함 O

# 값을 반복해서 벡터 만들기
np.repeat(a, repeats=3, axis=None) # 배열 a 를 3번 반복, axis는 축 (평평하게)

# 1. 단일 값 반복
repeated_vals = np.repeat(8, 4)

# 2. 배열 반복
repeated_arr = np.repeat([1, 2, 4], 2) # array([1, 1, 2, 2, 4, 4])

# 3. 각 요소의 반복 횟수 지정
repeated_each = np.repeat([1, 2, 4], repeats=[1, 3, 2])

# 4. 벡터 전체를 반복
repeated_whole = np.tile([1, 2, 4], 2)

# 벡터 길이 재는 방법
a = np.array([1, 2, 3, 4, 5])
len(a)
a.shape # 각 차원의 크기를 튜플 형태로 반환 (함수 X, 속성 O)
a.size # 전체 요소의 개수

# 2차원 배열
b = np.array([[1, 2], [3, 4]])
len(b) # 2 
b.size # 4
b.shape # 2


'''
[벡터 연산하기]
- 반복문 사용 X, 여러값 동시에 처리 O
- 가독성 향상, 성능 향상 가능
- 벡터 간 길이가 같아야 한다.
'''

# 벡터 생성
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 벡터 간 덧셈
add_result = a + b

# 벡터 간 뺄셈
sub_result = a - b

# 벡터, 상수 간 연산
x = np.array([1, 2, 4, 5])
y = x * 2



'''
[브로드캐스팅 : 길이가 다른 배열 간의 연산을 가능하게 해줌]
- 작은 배열이 큰 배열의 길이에 맞추어 자동 확장
- 끝 차원부터 시작해서 앞으로 진행
- 차원의 크기가 같거나, 차원 중 하나의 크기가 1인 경우
'''

# 길이가 다른 두 벡터
a = np.array([1, 2, 3, 4])
b = np.array([1, 2])
result = a + b  # Value 에러 발생
result = a + np.tile(b, 2) # tile 을 통해 길이 늘린 후 더해주면 가능


# 2차원 배열 생성
matrix = np.array([[0.0, 0.0, 0.0], 
                 [10.0, 10.0, 10.0], 
                 [20.0, 20.0, 20.0], 
                 [30.0, 30.0, 30.0]])
vector = np.array([1.0, 2.0, 3.0])
result = matrix + vector
## 브로드캐스팅 결과
## [[ 1. 2. 3.]
## [11. 12. 13.]
## [21. 22. 23.]
## [31. 32. 33.]]


# 벡터에 세로 벡터를 더하는 방법
matrix = np.array([[0.0, 0.0, 0.0], 
                 [10.0, 10.0, 10.0], 
                 [20.0, 20.0, 20.0], 
                 [30.0, 30.0, 30.0]])
vector = np.array([1.0, 2.0, 3.0, 4.0])
matrix.shape
vector.shape
result = matrix + vector # Value 에러 -> 세로 벡터로 변환해야 함

# 세로 벡터 생성
vector = vector.reshape(4, 1)
# array([[1.],
#        [2.],
#        [3.],
#        [4.]])
result = matrix + vector # 브로드캐스팅 가능

# 벡터 내적 : 두 벡터의 요소를 곱한 후 합산하는 연산
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot_product = np.dot(a, b) # 32

# 랜덤 함수
np.random.seed(2025) # 특정 seed 값에 따라 난수 생성
a = np.random.randint(1, 101, 20) # 1~20 까지 랜덤 숫자 10개
# array([ 7, 20, 15, 11,  8,  7, 19, 11, 11,  4], dtype=int32)

# 조건에 따른 필터링
a[a > 50]

'''
Q1
시드는 2025 고정
- 고객 데이터 만들기 (나이 20~80)
- 무작위 고객 3000명
- 40세 이상 고객 명 수 구하기
- 40대 고객 명 수 구하기
'''
np.random.seed(2025)
customer_age = np.random.randint(20, 81, 3000)
over_40_under_50 = customer_age[(customer_age >= 40) & (customer_age < 50)]
over_40_under_50.shape
over_40 = customer_age[customer_age >= 40]
over_40.shape


# 벡터 슬라이싱 예제
a = np.array([5, 3, 1, 10, 24, 3])
a > 5   # array([False, False, False,  True,  True, False])
a[a > 5]

# 논리 연산자 활용
a = np.array([True, True, False])
b = np.array([False, True, False])
a & b   # array([False,  True, False])
a | b   # array([True,  True, False])

# 필터링 + 논리 연산자
a = np.array([5, 3, 1, 10, 24, 3])
a[(a > 5) & (a < 15)]

# 필터링을 이용한 벡터 변경
a = np.array([5, 10, 15, 20, 25, 30])
a[a >= 10] = 10
a   # array([ 5,  3,  1, 10, 10,  3])

