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