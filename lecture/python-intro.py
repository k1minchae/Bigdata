import math
print("hello")

a = 3
a += 10
print(a)

x = 10
x /= 3
print(x, type(x))

a = 100
a += 10
a -= 20
a //= 2
print(a)

result = 10 + 2 * 3 ** 2 / 6 - (4 + 2) ** 2
print(result)

# True == 1, False == 0
# 나중에 벡터 연산을 할 때 연산을 빨리 하는 것이 중요함 -> 숫자로 바꿔서 계산

print(True and True)
print(True and False)
print(False and True)
print(False and False)
print(False * False) # and 는 곱셈과 같다

True or False
False or True
False + True
False + False
True or True # True
True + True # 2
min(1, True + True) # 1

# 멤버십 연산자
my_list = [1, 2, 3, 4, 5]
print(3 in my_list)

my_str = "Python programming"
"Python" in my_str

# 할당 연산자
x = [1, 2, 3]
y = [1, 2, 3]
x == y # True
x is y # False

z = x
x is z # True
y is z # False

my_list[2:4]


