import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import imageio

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
z = np.array([[1, 2], [3, 4], [5, 6]])
# 합계 구하기
z.sum(axis=0)   # 0: 열별 합계, 1: 행별 합계, 안 쓰면 전체합계

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


a = np.array([1, 2, 3, 4])
b = np.array([1, 2]).reshape(2, 1)  # 2차원 배열로 바뀜

# 브로드캐스팅은 차원이 작은 쪽(1차원) -> 큰 쪽으로 확장된다.
# a가 2차원으로 확장되면서 덧셈이 가능해짐

print(a + b)
print("=============")
print(b + a)
print("============")
print(a, b)

str_vec = np.array(["사과", "수박", "배", "참외"])
str_vec[[0, 2, 1, 0]]   # 여러개의 인덱스에 한번에 접근할 수 있다.

# 여러 벡터들 묶기
mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
combined_vec = np.concatenate((str_vec, mix_vec))
combined_vec

# column_stack() : 세로 쌓기
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))

# vstack() : 가로 쌓기
row_stacked = np.vstack((np.arange(1, 5), np.arange(12, 16)))
row_stacked


# 길이가 다른 벡터 합치기
a = np.arange(1, 5)
b = np.arange(12, 18)
a = np.resize(a, len(b))    # 강제로 길이 맞춰주기 (값을 앞에서부터 채워줌)
uneven_stacked = np.column_stack((a, b))


# select() : 여러개의 조건 처리
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
    # [array([ True, False,  True, False, False]),
    #  array([False, False, False, False,  True]),
    #  array([False,  True, False,  True, False])]

choices = ["양수", "0", "음수"]     # 조건과 결과 list 의 길이가 같아야함
result = np.select(conditions, choices, default="기타")
print(result)


# 고객 데이터 실습
np.random.seed(2025)
customer_size = 3000
age = np.random.randint(20, 81, customer_size)
gender = np.random.randint(0, 2, customer_size)
price = np.random.normal(50000, 3000, customer_size)

# 1: 고객 연령층 (2030, 4050, 6070, 80) 벡터 만들기
conditions = [(age >= 20) & (age < 40), # 2030
              (age >= 40) & (age < 60), # 4050
              (age >= 60) & (age < 80)] # 6070
choices = ["20-30대", "40-50대", "60-70대"]
age_group = np.select(conditions, choices, "80대 이상")


# 1에서 20까지 채워진 4행 5열을 만들려면?
np.arange(1, 21).reshape(4, 5)
np.arange(1, 21).reshape(6, 3)  # 부족해도 에러
np.arange(1, 21).reshape(5, 5)  # 남아도 에러

# np.arange(1, 21) 을 5, 5 행렬에 넣고싶은 경우는?
np.resize(np.arange(1, 21), 5 * 5).reshape(5, 5)

# 숫자를 채우는 방향 고르기
np.arange(1, 21).reshape(4, 5, order="C")   # 가로로 숫자를 채운다
np.arange(1, 21).reshape(4, 5, order="F")   # 세로로 숫자를 채운다

# 행렬 인덱싱
x = np.arange(1, 11).reshape((5, 2)) * 2
x[0, 1] # 행, 열 (0행 1열의 원소 반환)
x[:, 1] # 1열의 모든 원소 반환
x[[1, 2, 3], [1]]   # 1행, 2행, 3행에서 1열의 원소를 반환

y = np.arange(1, 21).reshape(4, 5)
y[2:4, 3]   # 2, 3행에서 3번째열에 있는 원소 반환
y[2, 1:4]
y[1:, 2:4]
y[1:3, [1, 3, 4]]

# 조건문 필터링
x[x[:, 1] > 15, 0]  # 1열의 원소가 15보다 큰 행의 0열의 원소 
y[y[:, 0] > 10, :]  # 0번째 열이 10보다 큰 모든 행의 모든 열 원소

# 1에서 20까지 숫자중 랜덤하게 20개의 숫자를 발생후 4행 5열 행렬 만드시오.
np.random.seed(2025)
z = np.random.randint(1, 21, 20).reshape(4, 5)
z
z[:, 0] > 7 # 참인 행: 0, 2
z[z[:, 0] > 7, :]   # 0행의 모든 원소, 2행의 모든 원소 필터링

z[2, :] > 10    # 참인 열: 3, 4
z[:, z[2, :] > 10]  # 전체 배열에서 3, 4열 필터링


y[0:2, 1].reshape(2, 1) # reshape 을 안 해도 입력 배열의 차원을 내부적으로 조정
y[2:, 3:]
np.column_stack((y[0:2, 1].reshape(2, 1), y[2:, 3:]))

y[:, 0] > 10 # 1차원 np 배열


'''
데이터 필터링 실습

'''
# 평균 점수 1등 학생의 점수 벡터 출력하기 (행: 학생, 열: 1~5월 모의고사 점수)
# 1: 행, 0: 열
z
z[z.mean(axis=1) == z.mean(axis=1).max(), :]


# 모의고사 평균 점수가 10점 이상인 학생들 데이터 필터링
z[z.mean(axis=1) >= 10, :]


# 모의고사 평균 점수가 10점 이상인 학생들 
# 3월 이후 모의고사 점수 데이터 필터링
z[z.mean(axis=1) >= 10, 2:]



a = np.arange(10)
a[2:6]
b = np.array([0, 2, 3, 6])
b > [4, 2, 1, 6]    # array([False, False,  True, False])


# 1~5월 모의고사 점수
# 기존 1월~4월 모의고사 점수 평균보다 
# 5월 모의고사를 잘 본 학생 데이터 필터링
z[z[:, :4].mean(axis=1) < z[:, 4], :]


'''
1~5월 모의고사 점수
기존 1월~4월 모의고사 점수 평균보다 
5월 모의고사 점수를 비교했을 때 
가장 점수가 많이 향상된 학생, 
가장 점수가 떨어진 학생의 
평균점수, 5월 모의고사 점수를 구하시오.
'''

# 5월 모의고사 점수 - 1~4월 평균점수
mean_diff = z[:, -1] - z[:, :-1].mean(axis=1) 

# 가장 많이 향상된 학생/가장 떨어진 학생 구하는 Boolean Array
best_student = mean_diff == mean_diff.max()   
worst_student = mean_diff == mean_diff.min()

print(f'{np.argmax(mean_diff) + 1}번째 학생의 평균 점수: {z[best_student, :].mean()}, 5월 모의고사 점수: {z[best_student][0][-1]}')
print(f'{np.argmin(mean_diff) + 1}번째 학생의 평균 점수: {z[worst_student, :].mean()}, 5월 모의고사 점수: {z[worst_student][0][-1]}')


# 사진은 행렬이다 (0: 검은색, 1: 흰색)
np.random.seed(2024)
# 데이터 생성 (예시: 3x3 행렬)
data = np.array([
 [0, 1, 1, 0, 0, 0, 1, 1, 0],
 [1, 1, 1, 1, 0, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 1, 1, 1, 1, 1, 1, 1, 1],
 [0, 1, 1, 1, 1, 1, 1, 1, 0],
 [0, 0, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 0, 1, 1, 1, 0, 0, 0]
])
img1 = np.random.rand(3, 3)
print("이미지 행렬 img1: \n", img1)
plt.figure(figsize=(10, 5))
plt.imshow(data, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


# DataFrame으로 변환 후 CSV 파일로 저장
df = pd.DataFrame(data, columns=["col1", "col2", "col3"])
df.to_csv("img_mat.csv", index=False)
print("img_mat.csv 파일이 생성되었습니다.")
img_mat = np.loadtxt('img_mat.csv', delimiter=',', skiprows=1)
img_mat


# 이미지 다운로드
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")    # urllib 을 통해 사진 다운

# 이미지 읽기
jelly = imageio.imread("jelly.png") # imageio 를 통해 사진 불러오기
type(jelly) # array
jelly.shape # (88, 50, 4: red, green, blue, alpha-투명도)
plt.imshow(jelly[:, :, 0], cmap='gray', interpolation='nearest')

jelly.max() # 255
jelly.min() # 0
jelly[:, :, 0].max()    # 239
jelly[:, :, 0].min()    # 0

plt.imshow(jelly / 255) # 0~1 로 normalize
plt.imshow(jelly)
plt.imshow(jelly[:, :, 0] / 255, cmap='gray', interpolation='nearest')    # 맨 앞면을 잘라온다.
plt.imshow(jelly[:, :, 1] / 255, cmap='gray', interpolation='nearest')    # 2번째 장을 잘라온다.
plt.imshow(jelly[:, :, 2] / 255, cmap='gray', interpolation='nearest')    # 3번째 장을 잘라온다.
plt.imshow(jelly[:, :, 3] / 255, cmap='gray', interpolation='nearest')    # 맨 뒷면을 잘라온다.


# 두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

# 3차원 배열로 합치기
my_array = np.array([mat1, mat2])
my_array.shape # (2, 2, 3)  
my_array[0, :, :]   # 0번째 2x3
my_array[1, :, :]   # 1번째 2x3
my_array[1, 1, 1:]

'''
행렬의 곱셈 : 앞 열 개수 == 뒷 행 개수 일때 가능 (2, 2) (2, 1) => 가운데 숫자가 같음.

결과 행렬의 모양: 앞 행의 개수, 뒷 열의 개수 (2, 2) (2, 1) => (2, 1)

'''

x = np.array([[2, 7], [1, 2]])
y = np.array([[0, 2], [3, 1]])
x.dot(y)


x = np.array([[1, 2, 3, 10], [7, 8, 9, 11], [4, 5, 6, 12]])
y = np.array([[0, 1, -1], [1, 2, 3], [0, 3, 1], [1, 4, 2]])
x.dot(y)



# 실습: 중간 고사 성적 데이터
np.random.seed(2025)
z = np.random.randint(1, 21, 20).reshape(4, 5)  # 각 학생별 국영수사과 성적

w = np.array([0.1, 0.2, 0.3, 0.1, 0.3])  # 각 과목에 대한 가중치
weighted_mean = z @ w    # 가중 평균
z.mean(axis=1)   # 그냥 평균

# 전부 값이 같다.
a = np.arange(1, 101)
a.dot(a)    # == a @ a
sum(a ** 2)

# matmul() 과 비교 : matmul은 딱 행렬의 곱셈만 됨 (브로드캐스트나 내적 이런건 X)
a
np.matmul(a, a) # 1D 벡터 x 1D 벡터인 경우만 특별히 내적으로 처리하도록 예외처리
b = np.arange(1, 101).reshape(1, 100)  # b.shape (100, 1) => 2D
np.matmul(b, b) # 이건 오류 !!


# 3차원 행렬의 곱셈
matC = np.random.rand(2, 3, 4)
matD = np.random.rand(2, 4, 5)
matC.shape, matD.shape  # 3차원 배열
np.matmul(matC, matD).shape # (2, 3, 5)


# 각 원소별 곱셈 (크기가 같은 경우)  !=  행렬의 곱셈
z = np.arange(10, 14).reshape((2, 2))
y = np.array([[1, 2], [3, 4]])
z * y


# 행렬의 역행렬 (행렬 세계에서의 역수)
'''
역행렬이 없는 경우
1. 정사각형이 아닌 경우
2. 선형 종속이 있는 경우

'''
no_inverse = np.array([[1, 2], [1, 2]]) # 선형 종속 O
np.linalg.inv(no_inverse)  # np.linalg.LinAlgError: Singular matrix

can_inverse = np.array([[1, 2], [3, 4]]) # 선형 종속 X
np.linalg.inv(can_inverse)
# array([[-2. ,  1. ],
#        [ 1.5, -0.5]])
np.matmul(can_inverse, np.linalg.inv(can_inverse))  # Identity matrix


a = np.array([1, 2, 2, 4]).reshape(2, 2)
np.linalg.inv(a)    # Singular matrix error
np.linalg.det(a)    # 0 이 나오면 inverse 가 존재 X

# 성적 데이터
# z^T z의 역행렬은 존재하나요?
np.random.seed(2025)
z = np.random.randint(1, 21, 20).reshape(4, 5)
z.transpose()
new = np.matmul(z.transpose(), z)
np.linalg.det(new)  # np.float64(-4.279112108578448e-06) => 부동소수점 오차 때문에 수치적으로 정확 X
inv_new = np.linalg.inv(new)
np.matmul(new, inv_new)


'''
1. n * p 행렬 X 에 대해서 X^T @ X는 항상 정사각형이다.
2. 데이터를 행렬 X로 보면, X에 가중치 벡터 W 를 곱했을 때, 결과 값은 각 데이터에 가중치를 곱한 것과 같다.
3. identity matrix (단위행렬: 대각 성분은 1, 나머지는 0) 은 곱셈에서의 1과 같다.
4. inverse matrix (역행렬) 는 곱셈에서의 역수와 같다. (A^-1 로 표현한다.)
5. 역행렬은 항상 존재하는 것은 아니다.
    - 정사각형 모양 행렬만 역행렬이 존재
    - np.linalg.det() 함수로 행렬식을 구할 수 있다. 
    - 행렬식값이 0이 아닐 때 역행렬이 존재한다. (non-singular matrix)

'''


# 연립방정식과 역행렬
# 3x + 3y = 1, 2x + 4y = 1을 행렬로 나타내면?
# A행렬 @ x,y 행렬 = B
a = np.array([[3, 3], [2, 4]])
np.linalg.det(a)    # 6 => 역행렬 O
inv_a = np.linalg.inv(a)
b = np.array([1, 1])
inv_a @ b   # array([0.16666667, 0.16666667])   => x: 0.16666667, y: 0.16666667

help(np.arange)
dir(np.arange)



np.random.seed(2025)
customer_size = 3000
gender = np.random.randint(0, 2, customer_size) # 0남, 1여
age = np.random.randint(20, 81, customer_size)
price = np.random.normal(50000, 3000, customer_size)

# 1: 고객 연령층 (2030, 4050, 6070, 80) 벡터 만들기
conditions = [(age >= 20) & (age < 40), # 2030
              (age >= 40) & (age < 60), # 4050
              (age >= 60) & (age < 80)] # 6070
choices = ["20-30대", "40-50대", "60-70대"]
age_group = np.select(conditions, choices, "80대 이상")

# price 변경: 구매액 여자인경우 + 1000 하기.
# gender 벡터를 쭉 훑어보다가 여자인경우 변경하고 남자인 경우에는 건너뛴다.
# while & continue 이용하기.
idx = -1
while idx < len(gender) - 1:
    idx += 1
    if gender[idx] == 0:
        continue
    price[idx] += 1000


# np 활용 : price[gender == 1] += 1000