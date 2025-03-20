# 범주형 데이터: 문자열을 사용한 데이터를 효율적으로 저장하는 방법
# 범주형 데이터를 사용하면 메모리 사용량이 줄어들고 연산속도가 빨라진다.

# 성별: 남자 & 여자 -> 0, 1로 변환
# 상품명 -> 상품 코드로 변환

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

values = pd.Series(["apple", "orange", "apple", "apple"] * 2)
values.unique() # 중복되지 않는 값만 출력
values.value_counts() # 각 값의 개수를 출력

# take 메서드
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(["apple", "orange", "banana"])
dim.take(values) # take 메서드를 사용하여 정수형 배열을 문자열로 변환
# 인덱스가 0인 값은 apple, 1인 값은 orange, 2인 값은 banana로 변환

fruits = ["apple", "orange", "apple", "apple"] * 2
n = len(fruits)
rng = np.random.default_rng(12345)
df = pd.DataFrame({"fruit": fruits,
                   "basket_id": np.arange(n),
                   "count": rng.integers(3, 15, size=n),
                   "weight": rng.uniform(0, 4, size=n)},
                  columns=["basket_id", "fruit", "count", "weight"])

# 범주형 변수 변환
fruit_cat = df['fruit'].astype('category')

# 카테고리 안에는 categories와 codes 속성이 있다.
# categories 속성은 카테고리 이름을 담고 있는 배열이고
# codes 속성은 정수로 인코딩된 카테고리 배열이다.
c = fruit_cat.array
c.categories
c.codes 

# 카테고리 객체를 사용하여 범주형 데이터를 변환
df['fruit'] = df['fruit'].astype('category')
df.info()

dict(enumerate(c.categories))   # 0은 apple, 1은 orange 로 매칭되어있음


# 범주형 변수를 Series 만들듯이 직접 생성할 수 있음
pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar']).codes


# 범주형 변수를 직접 생성 2
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
pd.Categorical.from_codes(codes, categories)


# 순서 부여
pd.Categorical.from_codes(codes, categories, ordered=True)


# 범주형 변수 연산
draws = rng.standard_normal(1000)

# 4개로 쪼개기
bins = pd.qcut(draws, 4, 
        labels=["Q1", "Q2", "Q3", "Q4"])

bins.categories
bins.codes[:5]


# 문제: 무게 변수를 기준으로 "가벼움, 중간, 무거움"
# weight_cat 변수 추가
df
label = ["가벼움", "중간", "무거움"]
bins = pd.qcut(df['weight'], 3, labels=label)
df['weight_cat'] = bins
df['weight_cat']


# 각 그룹의 weight 평균을 구해주세요
df.groupby('weight_cat')['weight'].mean()
df.pivot_table(values='weight',
               index="weight_cat")


# 범주형 칼럼이 메모리 작게 차지함
n = 10_000_000  # 가독성 (1000만)
labels = pd.Series(["foo", 'bar', 'baz', 'qux'] * (n // 4))
len(labels)     # n

# categories 속성은 4개, codes 속성은 10,000,000
labels_cat = labels.astype("category")
labels_cat

# 메모리 사용 60배 차이
labels_cat.memory_usage(deep=True)      # category O
labels.memory_usage(deep=True)          # category X



# 카테고리컬 변수 메서드
s = pd.Series(['a', 'b', 'c', 'd'] * 2)
cat_s = s.astype('category')

cat_s.cat.categories
cat_s.cat.codes



# 내가 현재 가지고 있는 변주형 변수 값이
# 전체 가질 수 있는 값이 아닌 경우
cat_s
actual_cat = ["a", "b", "c", "d", "e", "f"]

# categories 속성에 e, f가 추가되었음
cat_s2 = cat_s.cat.set_categories(actual_cat)


# 문제: cat_s의 첫번째 원소를 'e' 로 바꿔주세요
cat_s = cat_s.cat.add_categories('e')
cat_s[0] = 'e'

cat_s.value_counts()
cat_s2.value_counts()



# 더미 변수 생성법
pd.get_dummies(cat_s, dtype=int)
pd.get_dummies(cat_s, dtype=int, drop_first=True)


cat_var1 = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 1]
cat_var2 = pd.Categorical.from_codes(codes, categories)
cat_var1.value_counts()
cat_var2.value_counts()

plt.bar(cat_var1.categories, cat_var1.value_counts())
plt.bar(cat_var2.categories, cat_var2.value_counts())

cat_var1.value_counts().plot(
        kind='bar',
        title='Categorical Variable 1',
        figsize=(8, 4),
        )  # 점선(격자) 추가

cat_var2.value_counts().plot(
    kind='bar',
    title='Categorical Variable 1',
    figsize=(8, 4))