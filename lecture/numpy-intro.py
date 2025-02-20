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

# 1에서 300 사이 숫자들 중에서 7의 배수의 합을 구하세요
a = np.arange(1, 301)
result = a[a % 7 == 0].sum()

# 랜덤 함수에서 각 숫자가 나올 확률이 같은가?
np.random.seed(2025)
a = np.random.randint(1, 4, 3000)
a[a == 1].size, a[a == 2].size, a[a == 3].size
np.random.rand(1)   # 실수 버전

vec3 = np.random.rand(10 ** 9)      # 10억개로 테스트
vec3[vec3 > 0.5].size / 10 ** 9     # 표본이 커질수록 신뢰도가 높아짐


# 조건을 만족하는 위치 탐색
a = np.array([5, 3, 1, 10, 24, 3])
result = np.where(a < 7)    # (array([0, 1, 2, 5]),) index 반환

# np.where(조건, 참일 때 값, 거짓일 때 값)
result = np.where(a < 7, "7미만", "7이상")

# 필터링을 이용한 벡터 변경
a = np.array([5, 10, 15, 20, 25, 30])
a[a >= 10] = 10
a   # array([ 5,  3,  1, 10, 10,  3])

'''
[고객 데이터 만들기]
1. 3000명에 대해 성별 벡터를 만들어 보세요.
2. 0: 남자, 1: 여자
3. 50%, 50% 비율
4. 0과 1로 된 벡터를 "남자", "여자" 로 바꾸세요.
'''
np.random.seed(2025)
customer_size = 3000
gender = np.random.randint(0, 2, customer_size)
gender = np.where(gender == 0, "남자", "여자")
gender[gender == "남자"].size / customer_size, gender[gender == "여자"].size / customer_size

# 나이 벡터 생성 
age = np.random.randint(20, 81, customer_size)

# 나이 벡터에서 여자에 해당하는 나이들은 어떻게 걸러낼까?
age[gender == "여자"]   # Boolean Indexing

# 여자 고객 평균 나이
age[gender == "여자"].mean()    # np.array 의 내장 메서드
np.mean(age[gender == "여자"])   # numpy 전역 메서드
age[gender == "남자"].mean()

# 각 연령대별 평균 나이 계산해주세요 (80세는 70대로 설정)
age_20 = age[age < 30].mean()
age_30 = age[(age >= 30) & (age < 40)].mean()
age_40 = age[(age >= 40) & (age < 50)].mean() 
age_50 = age[(age >= 50) & (age < 60)].mean()
age_60 = age[(age >= 60) & (age < 70)].mean()
age_70 = age[(age >= 70)].mean()
age_20, age_30, age_40, age_50, age_60, age_70

# 성별 연령대별 평균 구매액 구하기 (2030, 4050, 6070)
price = np.random.normal(50000, 3000, customer_size)     # normal : 정규분포를 따르는 난수 생성 (평균, 표준편차, 데이터개수)
mean_price_2030_m = price[(age < 40) & (gender == "남자")].mean()
mean_price_2030_w = price[(age < 40) & (gender == "여자")].mean()
mean_price_4050_m = price[(age >= 40) & (age < 60) & (gender == "남자")].mean()
mean_price_4050_w = price[(age >= 40) & (age < 60) & (gender == "여자")].mean()
mean_price_6070_m = price[(age >= 60) & (gender == "남자")].mean()
mean_price_6070_w = price[(age >= 60) & (gender == "여자")].mean()

# 총 구매액
total_price_2030_m = price[(age < 40) & (gender == "남자")].sum()
total_price_2030_w = price[(age < 40) & (gender == "여자")].sum()
total_price_4050_m = price[(age >= 40) & (age < 60) & (gender == "남자")].sum()
total_price_4050_w = price[(age >= 40) & (age < 60) & (gender == "여자")].sum()
total_price_6070_m = price[(age >= 60) & (gender == "남자")].sum()
total_price_6070_w = price[(age >= 60) & (gender == "여자")].sum()

# 평균 구매액이 가장 높은 그룹은?
age_labels = np.array(["2030 남자", "2030 여자", "4050 남자", "4050 여자", "6070 남자", "6070 여자"])
mean_age_group = np.array([mean_price_2030_m, mean_price_2030_w, mean_price_4050_m, mean_price_4050_w, mean_price_6070_m, mean_price_6070_w])
total_age_group = np.array([total_price_2030_m, total_price_2030_w, total_price_4050_m, total_price_4050_w, total_price_6070_m, total_price_6070_w])
print("평균 구매액 최대 그룹: ", age_labels[np.argmax(mean_age_group)])     # argmax: 최대값의 인덱스를 알려줌
print("총 구매액 최대 그룹: ", age_labels[np.argmax(total_age_group)])     # argmax: 최대값의 인덱스를 알려줌


'''
벡터 함수 사용하기
'''
a = np.array([1, 2, 3, 4, 5])
sum_a = np.sum(a)
mean_a = np.mean(a)
median_a = np.median(a) # 중앙값 계산
std_a = np.std(a, ddof = 1) # 표준 편차, ddof: 자유도 (기본값: 0 -> 모집단)
var_a = np.var(a, ddof = 1) # 분산

# 데이터 타입 (기본값: float64)
a = np.array([1, 2, 3], dtype=np.int32)
a = np.array([1.5, 2, 3], dtype=np.int32)   # array([1, 2, 3], dtype=int32)
a = np.array([1.5, 2, 3], dtype=np.float64) # array([1.5, 2. , 3. ])
a = np.array([1.5, 0, 3], dtype=np.bool_)   # array([ True, False,  True])

gender = np.random.randint(0, 2, 3000)
gender = np.array(gender, dtype=np.str_)
gender[gender == "0"] = "남자"
gender[gender == "1"] = "여자"

# nan : not a number (float 타입)
a = np.array([20, np.nan, 13, 24, 309])
np.mean(a)      # nan
np.nanmean(a)   # 91.5
a_filtered = a[~np.isnan(a)]    # nan이 생략된 벡터 만들기
b = np.array([20, None, 13, 24, 309])  # None 은 연산자체가 불가능

a = np.array([20, np.nan, 13, 24, 309])
a[np.isnan(a)] = np.nanmean(a)  # nan 값을 평균값으로 대체


