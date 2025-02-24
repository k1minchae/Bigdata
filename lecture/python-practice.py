# 문제 1 : 리스트에서 최대 곱 계산
numbers = [1, 10, 3, -5, 7, -10]
n = len(numbers)
max_val = float('-inf')
for i in range(n):
    for j in range(n):
        if i != j:
            max_val = max(numbers[i] * numbers[j])
print(max_val)


# 문제 2 : 비트 연산으로 짝수/홀수 확인
numbers = [12, 17, 24, 33, 42, 51]
result = {}
for num in numbers:
    if num & 1:
        result[num] = "odd"
    else:
        result[num] = "even"


# 문제 3 : 문자열에서 중복 문자 제거
string = "programming"
result = ""
visited = {}
for s in string:
    if not visited.get(s, False):
        result += s
        visited[s] = True
print(result)


# 문제 4 : 2차원 리스트에서 최대값 찾기

matrix = [[3, 5, 7], [1, 6, 9], [8, 4, 2]]
max_val = float('-inf')
max_pos = (-1, -1)
n = len(matrix)
m = len(matrix[0])

for i in range(n):
    for j in range(m):
        if matrix[i][j] > max_val:
            max_val = matrix[i][j]
            max_pos = (i, j)

print(max_val, max_pos)


# 문제 5 : 멤버십 연산자와 딕셔너리 활용
scores = {"Alice": 85, "Bob": 90, "Charlie": 78, "Dina": 92}
names_to_check = ["Alice", "Bob", "Eva", "Charlie", "Zoe"]
sum_scores = 0
max_score = 0
max_student = ""

for name in names_to_check:
    if scores.get(name, -1) == -1: # 이름 X
        print(f"Student [{name}] is not found")
    else:
        print(f"Student [{name}] : {scores[name]}")
        sum_scores += scores[name]
        if scores[name] > max_score:
            max_score = scores[name]
            max_student = name

over_80 = []
for name, score in scores.items():
    if score >= 80:
        over_80.append(name)

print("80점 넘는 사람들: ", over_80)
print("존재하는 학생들 점수 합 : ", sum_scores)
print("존재하는 학생들 중 최고점: ", max_score)

# 문제 1 : 리스트 생성 및 기본 연산
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1.extend(list2)
repeated = combined * 2
sum_repeated = sum(repeated)

# 문제 2 : 리스트 인덱싱 슬라이싱
fruits = ["apple", "banana", "cherry", "date", "elderberry"]
print(fruits[0], fruits[-1])
subset = fruits[1:4]

# 문제 3 : 리스트 메서드 활용
fruits = []
fruits.append("apple")
fruits.append("banana")
fruits.append("cherry")
fruits[1] = "blueberry"
print(fruits)
fruits.pop()
print(fruits)
while fruits:
    fruits.pop()
print(fruits)

# 문제 4 : 리스트 내포 활용
even_squares = [x ** 2 for x in range(1, 11) if x % 2 == 0]
print(even_squares)

# 문제 5 : 다차원 리스트
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(matrix[0][1])
print(matrix[1])

# 문제 6 : 고객 데이터 분석
customers = [
    {"name" : "Alice", "age": 25}, 
    {"name" : "Bob", "age": 34}, 
    {"name" : "Charlie", "age": 29}, 
    {"name" : "David", "age": 32}, 
    {"name" : "Eve", "age": 22}, 
]
new_customer = [] # 나이 1년씩 증가
customer_over_30 = []
longest_name = ""
max_len = 0
sum_age = 0
for customer in customers:
    name = customer.get("name")
    age = customer.get("age")
    new_customer.append({"name" : name, "age": age + 1})
    if age >= 30:
        customer_over_30.append({"name": name, "age": age})
    if age < 30 and name[0] == "A":
        print("30 세 미만 고객: ", name)
    if max_len < len(name):
        max_len = len(name)
        longest_name = name
    sum_age += age
print("나이 1살 더한 고객들 : ", new_customer)
print("30세 이상 고객들: ", customer_over_30)
print("가장 이름이 긴 고객: ", longest_name)
print("고객 나이 합계 : ", sum_age)

# 문제 7: 판매 데이터 분석

sales_data = {
    "January": 90,"February": 110,
    "March": 95,"April": 120,
    "May": 80,"June": 105,
    "July": 135,"August": 70,
    "September": 85,"October": 150,
    "November": 125,"December": 95
}
sum_sales = 0
max_sales = 0
top_month = 0
for eng, data in sales_data.items():
    if data >= 100:
        print("판매량 100 이상인 월: ", eng)
    sum_sales += data
    if max_sales < data:
        max_sales = data
        top_month = eng
print("연간 총 판매량: ", sum_sales)
print("연 평균 판매량 : ", sum_sales / 12)
print("판매량이 가장 높은 월 : ", top_month, " 판매량: ", max_sales)

# 문제 8 : 제품 리뷰 분석
reviews = [
    "This product is excellent and very useful",
    "The quality is good but not excellent",
    "Poor design but positive features",
    "Absolutely positive and worth the price",
    "Excellent choice for anyone looking for quality"
]

positive_review = 0
word_review = []
over_5 = []
cnt = {}
max_cnt = 0
max_word = ""
for review in reviews:
    if "positive" in review:
        positive_review += 1
    if "excellent" in review:
        print(review.upper())
    temp = list(review.split())
    word_review.append(temp)
    over_5_temp = []
    for t in temp:
        cnt[t] = cnt.get(t, 0) + 1
        if cnt[t] > max_cnt:
            max_cnt = cnt[t]
            max_word = t
        if len(t) >= 5:
            over_5_temp.append(t)
    over_5.append(over_5_temp)

print("긍정적인 리뷰: ", positive_review)
print("단어 단위로 분리한 리뷰 list: ", word_review)
print("5글자 이상만 모은 리뷰 list: ", over_5)
print("가장 많이 등장한 단어: ", max_word, f" 는 {max_cnt} 번 등장")

# list comprehension 을 이용한 코드로 refactoring
a = [review.upper() for review in reviews if "EXCELLENT" in review.upper()]

# 문제 9 : 학생 성적 분석
grades = {
    "Alice": {"math": 95, "english": 88, "science": 92},
    "Bob": {"math": 72, "english": 75, "science": 68},
    "Charlie": {"math": 88, "english": 85, "science": 90},
    "Diana": {"math": 82, "english": 89, "science": 84}
}
over_90 = []
best_student = {}
best_score = {}
avg_scores = []
# 1. 각 학생의 평균 점수를 계산하고 출력하세요
for student, classes in grades.items():
    sum_score = 0
    class_cnt = 0
    avg_score = 0
    for subject, score in classes.items():
        class_cnt += 1
        sum_score += score
        if best_score.get(subject, 0) < score:
            best_score[subject] = score
            best_student[subject] = student
    avg_score = round((sum_score / class_cnt), 2)
    print(f"{student}학생의 평균점수 : {avg_score}")
    avg_scores.append(avg_score)

# 2. 평균 점수가90점 이상인 학생을 필터링하세요.
    if sum_score / class_cnt >= 90:
        over_90.append({student : avg_score})
print("평균점수 90점 이상인 학생들: ", over_90)

# 3 모든 학생의 과목별 최고 점수를 출력하세요
print(best_score)

# 4 모든 학생의 평균 점수를 기준으로 내림차순 정렬된 리스트를 생성하세요
avg_scores.sort(reverse=True)
print("평균점수 내림차순: ", avg_scores)

# 5 특정 과목 math의 최고 점수를 가진 학생의 이름을 출력하세요
print("math 의 최고점수인 학생: ", best_student.get("math", "null"))



'''
연습문제 1: 함수 정의와 기본값 설정
1. 함수 이름: add_numbers
2. 입력값: 두개의 숫자 a, b (기본값: a=1, b=2)
3. 출력값: 두 숫자의 합

기본값을 사용한 결과와 a = 5, b = 7일 때 결과를 각각 출력하세요.
'''
def add_numbers(a=1, b=2):
    return a + b

print(add_numbers())  # 3
print(add_numbers(5, 7))  # 12


'''
연습문제 2: 조건문 사용
1. 함수이름: check_sign
2. 입력값: 하나의 숫자 x
3. 출력값: x가 양수면 "양수", 음수면 "음수", 0이면 "0"

숫자 10, -5, 0에 대해 함수를 호출하고 결과를 출력하세요
예시: 10: 양수
'''
def check_sign(x):
    if x > 0:
        return f"{x}: 양수"
    elif x < 0:
        return f"{x}: 음수"
    else:
        return f"{x}: 0"

print(check_sign(10))
print(check_sign(-5))
print(check_sign(0))


'''
연습문제 3: 반복문 사용

1. 1부터10까지 숫자를 출력하는 함수를 작성하세요
- 함수 이름: print_numbers
- 출력값: 1부터10까지의 숫자를 줄바꿈하여 출력

2. 함수를 호출하여 결과를 확인하세요

'''
def print_numbers():
    for i in range(1, 11):
        print(i)
print_numbers()


'''
연습문제 4: 중첩 함수 사용

1. 다음 요구사항에 맞는 함수를 작성하세요
- 함수 이름: outer_function
- 내부에 inner_function을 정의하고 inner_function은 입력값에2를 더한 값을 반환
- outer_function은 inner_function을 호출하여 결과를 반환

2. 숫자5를 입력값으로 outer_function을 호출하고 결과를 출력하세요

'''

def outer_function():
    def inner_function(x):
        return x + 2
    return inner_function(5)

print(outer_function())  # 7


'''
연습문제 5: while 반복문과 break

1. 다음 요구사항에 맞는 함수를 작성하세요
- 함수 이름: find_even
- 입력값: 시작 숫자 start
- 동작: start부터 시작하여 처음으로 나오는 짝수를 반환
- 짝수를 찾으면 break로 반복문 종료

2. 함수에 start = 3을 입력하여 호출하고 결과를 출력하세요

'''
def find_even(start):
    while True:
        if start % 2 == 0:
            return start
        start += 1

find_even(3)  # 4