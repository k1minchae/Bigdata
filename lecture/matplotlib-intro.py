# 시각화
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4])
plt.ylabel('Some Numbers')
plt.show()

'''
plot에 단일 리스트 1234를 넣으면 Matplotlib이 이를 y값으로 해석하고 
자동으로 x값을 0123으로 설정합니다
'''

plt.plot([1, 2, 3, 4], [1, 4, 9, 16])   # plot(x, y)
# (1, 1), (2, 4), (3, 9), (4, 16)
plt.xlabel("This is number")
plt.ylabel("numbers")
plt.show()

# 넘파이 벡터가 plot() 작동되는지?
# y = x ** 3 그래프를 그리고 싶다.

x = np.arange(-10, 11, 1)
y = x ** 3
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x, y, 'ro')    # r: 빨강, o: 원형
plt.axis([0, 6, 0, 20]) # 0, 6: x축 / 0, 20: y축
plt.show()


x = np.arange(0., 5., 0.2) # 0~5 사이 0.2 간
plt.plot(x, x, 'r--', # 빨간색 점선
        x, x**2, 'bs', # 파란색 정사각형 마커
        x, x**3, 'g^') # 초록색 삼각형 마커



# 펭귄 데이터 불러오기
# 부리 길이, 부리 깊이, x, y 순서쌍으로 표현 + 점찍기
df = pd.read_csv('../practice-data/penguins.csv')

x = df['bill_length_mm']
y = df['bill_depth_mm']

plt.plot(x, y, 'ro')
plt.xlabel('bill length')
plt.ylabel('bill depth')
plt.show()


# numpy 데이터 바로 플로팅하기
my_data = {'my_x': np.arange(50),
        'my_y': np.random.randn(50) * 10}
pd.DataFrame(my_data)

# 점으로 표현되는 함수 (x축 칼럼이름, y축 칼럼이름)
plt.scatter('my_x', 'my_y', data=my_data) # x, y를 직접 플로팅

# 날개길이, 몸무게
plt.scatter('flipper_length_mm', 'body_mass_g', data=df)
plt.scatter('body_mass_g', 'flipper_length_mm', data=df)    # 비례관계


names = ['A', 'B', 'C']
values = [1, 10, 100]   # 높이
plt.figure(figsize=(9, 3))  # 가로 9, 세로 3
plt.title("category plotting")
plt.bar(names, values) # 막대 그래프
plt.show()

names = ['A', 'B', 'C']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(231)    # 2행 3열중 4번째
plt.bar(names, values) # 막대 그래프

plt.subplot(132)    # 2행 3열중 두번째
plt.scatter(names, values) # 산점도

plt.subplot(233)    # 1행 3열중 세번째
plt.plot(names, values) # 선 그래프
plt.suptitle('Categorical Plotting')
plt.show()


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t = np.arange(0., 5., 0.1)
t2 = np.arange(0., 5., 0.02)
plt.figure()
plt.subplot(211)
plt.plot(t, f(t), 'bo',
         t2, f(t2), 'k')
plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# 로그 축도 설정 가능
plt.xscale('log')
plt.yscale('log')
plt.show()


# 그래프에 주석 추가
plt.plot([1, 2, 3, 4], [10, 20, 30, 40])
# 해당 좌표에 주석 추가
plt.text(2, 25, 'Important Point', fontsize=12, color='red')
plt.show()


# 그래프 범례
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], label="y = x^2")  # 그래프 라벨 설정
plt.title("Example Plot")  # 그래프 제목 추가
plt.xlabel("X Axis")  # x축 라벨 추가
plt.ylabel("Y Axis")  # y축 라벨 추가
plt.legend(loc="upper left")  # 범례 추가 (네 귀퉁이에)
plt.show()


# 시각화 실습
df = pd.read_csv('../practice-data/Obesity2.csv')
df.head()
'''
- Gender : 성별
- Age : 나이
- Height : 키
- Weight : 몸무게
- SMOKE : 흡연 여부
- NObeyesdad : 비만 수준
    - overweight_level_i : 과체중 수준 I
    - obesity_type : 비만 유형(I ~ III)
'''

# 히스토그램 그리기

# bin: 상자 개수 (몇개를 그릴것인가?)
plt.hist(df[['Age']], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
# 사용 목적: 분포 형태 확인, 이상치 탐색

# 이상치 판단 후 삭제
df['Age'].isna().sum()
(df['Age'] >= 100).sum()     # 5명
df.loc[df['Age'] >= 100, :] # 150 살로 되어있네

# Nan 값으로 대체후 삭제
df.loc[df['Age'] >= 100, :] = np.nan
df.dropna()

# 해당 인덱스들 삭제
df = df.drop(df.loc[df['Age'] >= 100, :].index)

# filtering 으로 삭제
filtered_df = df.loc[~(df['Age'] >= 100), :]

plt.hist(df[['Age']], bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# bin 개수 정하는 룰 (scott's Rule)
bin_w = 3.5 * np.std(df['Age']) / (len(df['Age']) ** (1/3))   # 빈의 너비
bin_cnt = int((max(df['Age']) - min(df['Age'])) / bin_w)    # 20

# histogram
df["Age"].plot(kind="hist",  # Age(나이) 변수에 대한 히스토그램을 그림
                bins=10,  # 데이터를 10개의 구간(bin)으로 나눔
                edgecolor="black",  # 각 막대의 테두리 색상을 검은색으로 설정
                alpha=0.7,  # 막대의 투명도를 0.7로 설정 (0=완전 투명, 1=완전 불투명)
                figsize=(8, 5))  # 그래프 크기를 가로 8, 세로 5 인치로 설정
plt.xlabel("Age")  # x축 라벨을 "Age"로 설정
plt.ylabel("Frequency")  # y축 라벨을 "Frequency"로 설정
plt.title("Histogram of Age")  # 그래프 제목을 "Histogram of Age"로 설정
plt.text(20, 200, "fsdfsdfdsfdsfsd")
plt.show()  # 그래프를 화면에 출력


# 기본적인 밀도 곡선은 seaborn 라이브러리를 이용
import seaborn as sns
sns.kdeplot(df['Age'], shade=True)
plt.xlabel("Age")
plt.ylabel("Density")
plt.show()

# Bandwith : 작을수록 세밀한 분포 표시
sns.kdeplot(df['Age'], bw_method=0.1, shade=True)
sns.kdeplot(df['Age'], bw_method=0.5, shade=True)
plt.xlabel("Age")
plt.ylabel("Density")
plt.legend(["bw=0.1", "bw=0.5"])
plt.show()


# Pandas를 활용한 밀도 곡선
df['Age'].plot(kind='kde', figsize=(8,5))
plt.xlabel("Age")
plt.ylabel("Density")
plt.title("Density Plot of Age")
plt.show()

# 비차트
# 범주형 데이터의 빈도 계산
category_cnt = df["NObeyesdad"].value_counts()
category_cnt.index # x축
category_cnt.values # y축

plt.figure(figsize=(6, 5))
plt.bar(category_cnt.index, category_cnt.values,
        color="skyblue", alpha=0.7, edgecolor='black')
plt.xlabel("obesity level")
plt.ylabel("count")
plt.title("bar chart of obesity levels")
plt.xticks(rotation=45) # 가독성을 위해 x축 라벨 회전


# y축 보여주는 범위 변경 (250-~350)
plt.figure(figsize=(6, 5))
plt.bar(category_cnt.index, category_cnt.values,
        color="skyblue", alpha=0.7, edgecolor='black')
plt.xlabel("obesity level")
plt.ylabel("count")
plt.ylim(250, 350)      # y축 limit 설정


# Seaborn 이용한 막대형 차트
sns.barplot(x=category_cnt.index, y=category_cnt.values, palette='Blues_r')
plt.ylim(250, 350)      # y축 limit 설정
plt.title("obesity level")


# box plot
# 데이터의 분포를 약속된 그림을 통해 나타내주는 것
# 사분위(데이터의 분포를 4등분), 이상치는 점으로 표시

# 숫자를 오름차순 정렬 + 중앙값 찾기

# 상자그림(Box Plot) 그리기
plt.figure(figsize=(6,5))
sns.boxplot(x=df['NObeyesdad'], 
            y=df['Weight'], 
            palette="Pastel1")
plt.xlabel("Obesity Level")
plt.ylabel("Weight")
plt.title("Box Plot of Weight by Obesity Level")
plt.xticks(rotation=45)
plt.show()

'''
이상치 판단기준
1. IQR = Q3 - Q1 (상자의 길이)
2. 데이터 중에서 Q1과 Q3에서 1.5 * IQR 이상 떨어진 데이터는 이상치로 판단
'''


# pandas 를 활용한 상자그림 시각화
df.boxplot(column='Weight', 
           by='NObeyesdad', 
           grid=False, 
           figsize=(8, 5))  
plt.xlabel("Obesity Level")
plt.ylabel("Weight")
plt.title("Box Plot of Weight by Obesity Level")
plt.suptitle("")  # 자동 생성되는 그룹 제목 제거
plt.xticks(rotation=45)
plt.show()


np.random.seed(25317)
d = np.random.randint(1, 21, size=15)

# [ 1,  3,  3,  5,  5,  6,  6,  8, 11, 13, 13, 15, 18, 18, 20]
d.sort()
Q2 = 8
d[d < Q2] # [1, 3, 3, 5, 5, 6, 6]
Q1 = 5
d[d > Q2] # [11, 13, 13, 15, 18, 18, 20]
Q3 = 15

plt.boxplot(d)
plt.xlabel("data")
plt.ylabel("value")
plt.title("Box Plot of Random Int")


# 이상치가 있는 상자그림
np.random.seed(957)
y = np.random.randint(1, 21, size=15)
y[3] = 40
plt.boxplot(y)
plt.xlabel("data")
plt.ylabel("value")
plt.title("Box Plot with outlier")


# 비만 종류별 데이터 포인트 색깔을 변경해보세요.
df
plt.figure(figsize=(6, 5))
NObeyesdad = df['NObeyesdad'].unique()
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]
for i, ob in enumerate(NObeyesdad):
    temp = df.loc[df['NObeyesdad'] == ob, :]
    plt.scatter(temp['Height'], temp['Weight'], alpha=0.3, color=colors[i])
plt.legend(NObeyesdad)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot of Height and Weight by Obesity Level")
plt.show()




# np select 활용
# 고유한 NObeyesdad 값 가져오기
NObeyesdad = df['NObeyesdad'].unique()

# 색상 리스트 (고유한 obesity level과 매칭)
colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]

# np.select()를 사용하여 각 NObeyesdad 값에 해당하는 색상 선택
conditions = [df['NObeyesdad'] == ob for ob in NObeyesdad]
# default는 검정색 (혹시 없는 값이 있을 경우 대비)
df['color'] = np.select(conditions, colors, default="#000000")  

# 산점도 그리기
plt.scatter(df['Height'], df['Weight'], alpha=0.3, color=df['color'])
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=ob) for i, ob in enumerate(NObeyesdad)]
plt.legend(handles, NObeyesdad, title="Obesity Level")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Scatter Plot of Height and Weight by Obesity Level")


# sns 산점도
import seaborn as sns
# 이게훨씬편함!!!
sns.scatterplot(x='Height', y='Weight', hue='NObeyesdad', data=df, alpha=0.7)
sns.scatterplot(x='Weight', y='Age', data=df, alpha=0.7)
plt.xlabel("Weight")
plt.ylabel("Age")
sns.scatterplot(x='Age', y='Weight', data=df, alpha=0.7)
plt.xlabel("Age")
plt.ylabel("Weight")



# 산점도
# 두 변수의 상관관계를 파악
# 선형/비선형 관계 분석

# 키와 몸무게의 관계 + 각 종별 분석
penguin = pd.read_csv('../practice-data/penguins.csv')
penguin['species'].unique() # ['Adelie', 'Chinstrap', 'Gentoo']
adelie = penguin.loc[penguin['species'] == 'Adelie', :]
chinstrap = penguin.loc[penguin['species'] == 'Chinstrap', :]
gentoo = penguin.loc[penguin['species'] == 'Gentoo', :]
plt.scatter(adelie['body_mass_g'], adelie['flipper_length_mm'], alpha=0.5)
plt.scatter(gentoo['body_mass_g'], gentoo['flipper_length_mm'], alpha=0.5)
plt.scatter(chinstrap['body_mass_g'], chinstrap['flipper_length_mm'], alpha=0.5)
plt.xlabel("body mass")
plt.ylabel("flipper length")
plt.title("Scatter Plot of Body Mass and Flipper Length")
plt.legend(['Adelie', 'Gentoo', 'Chinstrap'])


# 히트맵 (Heatmap)
# 데이터 행렬을 색상으로 표현한 그래프
# 진할수록 큰 값
# 상관관계 분석에 사용

# 상관행렬 계산
corr_matrix = df[['Age', 'Height', 'Weight']].corr()
# 히트맵 그리기
plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, 
            annot=True,    # 그래프 글씨
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.show()


# 시계열 라인 그래프
# X축이 시간, Y축이 해당 시간의 값

dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
values = np.cumsum(np.random.randn(30)) + 50
df_timeseries = pd.DataFrame({"Date": dates, "Value": values})
plt.figure(figsize=(8,5))
plt.plot(df_timeseries['Date'], df_timeseries['Value'], marker='o', linestyle='-')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Line Graph")
plt.xticks(rotation=45)
plt.show()


# 모자이크 그래프
# 범주형 데이터의 빈도를 나타내는 그래프
# 사각형의 크기가 데이터의 빈도를 나타냄
from statsmodels.graphics.mosaicplot import mosaic

# 모자이크 그래프 그리기
plt.figure(figsize=(8,5))
mosaic(df, ['Gender', 'NObeyesdad'], title="Mosaic Plot of Gender vs Obesity Level")
plt.show()
