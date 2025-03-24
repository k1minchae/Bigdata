from math import factorial
from itertools import combinations, permutations
# [연습문제] 순열과 조합
# 문제 1.
# 서점에 5권의 서로 다른 책이 있습니다. 이 중 3권을 뽑아 순서 있게 진열하려면 몇 가지 방법이 있나요?
result = factorial(5) // factorial(2)
print(f'1번: {result}')


# 문제 2.
# 4명의 학생이 4개의 의자에 각각 앉는 경우의 수는?
result = factorial(4)
print(f'2번: {result}')


# 문제 3.
# 0, 1, 2, 3 중에서 숫자 3개를 골라 서로 다른 세 자리 수를 만들 수 있는 경우의 수는?
result = 3 * 3 * 2
print(f'3번: {result}')


# 문제 4.
# 문자 A, A, B, B로 만들 수 있는 서로 다른 순열은 몇 개인가요?
result = factorial(4) // (factorial(2) * factorial(2))
arr = ['a', 'a', 'b', 'b']
len(set(permutations(arr, 4)))
print(f'4번: {result}')



# 문제 5.
# 단어 LEVEL의 문자를 재배열할 수 있는 서로 다른 순열의 수는?
result = factorial(5) // (factorial(2) * factorial(2) * factorial(1))
arr = list(['l', 'e', 'v', 'e', 'l'])
len(set(permutations(arr, 5)))
print(f'5번: {result}')


# 문제 6.
# 단어 MISSISSIPPI의 문자를 재배열할 수 있는 서로 다른 순열의 수는?
result = factorial(11) // (factorial(4) * factorial(4) * factorial(2) * factorial(1))
print(f'6번: {result}')


# 문제 7.
# 6명 중에서 2명을 뽑아 팀을 만들려고 합니다. 몇 가지 방법이 있나요?
result = factorial(6) // (factorial(2) * factorial(4))
print(f'7번: {result}')


# 문제 8.
# 10개의 공 중에서 3개를 선택하려고 합니다. 순서를 고려하지 않고, 같은 공이 없을 때 경우의 수는?
result = factorial(10) // (factorial(7) * factorial(3))
print(f'8번: {result}')


# 문제 9.
# 10명으로 이루어진 학급에서 4명을 뽑아 회장, 부회장, 서기, 회계를 맡기려고 합니다. 몇 가지 방법이 있나요?
factorial(10) // factorial(10 - 4)
print(f'9번: {result}')


# 문제 10.
# 8개의 서로 다른 색 구슬 중에서 3개를 순서 없이 뽑는 방법은 몇 가지인가요?
result = factorial(8) // (factorial(3) * factorial(5))
print(f'10번: {result}')
