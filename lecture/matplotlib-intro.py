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



# flight 데이터 실습
from nycflights13 import flights, planes
flights.info()
planes.info()
flights.head()
planes.head()

flights['carrier'].unique()

# merge 사용해서 flights 와 planes 병합한 데이터로
# 각 데이터 변수 최소 하나씩 선택 후 분석
# 날짜 시간 전처리 코드 들어갈 것
# 문자열 전처리 코드 들어갈 것
# 시각화 종류 최소 3개 (배우지 않은것도 OK)

planes['type'].unique() # 'Fixed wing multi engine', 'Fixed wing single engine','Rotorcraft'
planes['engine'].unique()
planes['manufacturer'].unique()
planes['model'].unique()   # 127
planes['speed'].isna().sum() # 거의다 Nan값

data = pd.merge(flights, planes, on='tailnum', how='left')
data.head()


# 각 비행기 model별로 생산 년도 시각화 year
one_engine = data.loc[data['engines'] == 1, 'year_y']
two_engine = data.loc[data['engines'] == 2, 'year_y']
three_engine = data.loc[data['engines'] == 3, 'year_y']
four_engine = data.loc[data['engines'] == 4, 'year_y']


# 생산년도에따라 선호하는 엔진의 개수가 달라지는 것을 알 수 있다.
plt.figure(figsize=(12, 5))
sns.kdeplot(one_engine, bw_method=0.4, shade=True)
sns.kdeplot(two_engine, bw_method=0.4, shade=True)
sns.kdeplot(three_engine, bw_method=0.4, shade=True)
sns.kdeplot(four_engine, bw_method=0.4, shade=True)
plt.xlabel("Production year")
plt.legend(["1 engine", "2 engines", "3 engines", "4 engines"])
plt.title("Production year by Engine cnt")


# seats 수에 따른 지연시간




# 항공사별 좌석 수 (대중적인 항공사가 뭔지?)
seats_by_carrier = data.groupby('carrier')['seats'].sum().reset_index()

# 항공사 코드로 이름 매칭
airline_names = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "EV": "ExpressJet Airlines",
    "F9": "Frontier Airlines",
    "FL": "AirTran Airways",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "US": "US Airways",
    "VX": "Virgin America",
    "WN": "Southwest Airlines",
    "YV": "Mesa Airlines"
}

def replace_airline_code(row):
    row[0] = airline_names.get(row[0], "Unknown")  # carrier 코드 -> 항공사 이름 변환
    return row
airline_seats_data = np.apply_along_axis(replace_airline_code, axis=1, arr=seats_by_carrier)

plt.figure(figsize=(12, 5))
plt.bar(airline_seats_data[:, 0], airline_seats_data[:, 1])
plt.ylabel('seats')
plt.xlabel('Airlines')
plt.xticks(rotation=45)
plt.title('Popular Airline Info')



nycflights = flights

# 시간대 별 항공편 수 분석
# 시간대별로 지연 시간이 얼마나 달라지는지?
def divide_hour(hour):
    if 6 <= hour < 12:
        return 'morning'
    if 12 <= hour < 18:
        return 'lunch'
    if 18 <= hour < 24:
        return 'dinner'
    return 'dawn'

nycflights['time_of_day'] = nycflights['hour'].apply(divide_hour)

# 2-1) 공항별, 시간대 별로 항공편수가 몇개있는지
time_flights = nycflights.groupby(['time_of_day']).size()
plt.bar(['dawn', 'morning', 'lunch', 'dinner'], time_flights.values[[0, 3, 2, 1]])
plt.xlabel('time')
plt.ylabel('flights')
plt.title('flights by time')


# 15분 이상 지연된 비행기들
delayed_flights = nycflights.loc[nycflights['dep_delay'] >= 15, :]

# 지연된 비행기 시간대별로 분류
delayed_flight_cnt = delayed_flights.groupby('time_of_day').size()
plt.bar(['dawn', 'morning', 'lunch', 'dinner'], delayed_flight_cnt.values[[0, 3, 2, 1]])
plt.xlabel('time')
plt.ylabel('delayed flights')
plt.title('delay by time')

'''
- 항공편 수가 많은 아침에 가장 많은 지연을 예상했으나 
  아침보다 점심이, 그리고 저녁이 확연히 지연비율이 높음.
  
- Q) 앞에 항공편이 지연되는 것이 뒷 항공편에 영향을 미쳐서 
     항공편이 적음에도 저녁시간에 많은 지연이 발생되는것이 아닐까?
'''

# 연쇄지연 여부 분석
# 출발 시간 기준으로 정렬
sorted_flight = nycflights.sort_values(['year','month', 'day', 'dep_time'], ascending=True)
sorted_flight = sorted_flight.fillna(0)

# 같은 날, 이전 시간에 출발한 항공편의 도착 지연 정보 추가
sorted_flight['prev_arr_delay'] = sorted_flight.groupby(['year', 'month', 'day'])['arr_delay'].shift(1)
sorted_flight['prev_arr_delay']

# 연쇄 지연 여부 분석 (이전 항공편의 도착 지연이 현재 항공편의 출발 지연에 영향을 주었는지)
delay_cnt = len(sorted_flight.loc[sorted_flight['dep_delay'] >= 15, :]) # 72914
next_delay_cnt = len(sorted_flight.loc[(sorted_flight['dep_delay'] >= 15) & (sorted_flight['prev_arr_delay'] >= 15), :]) # 30797

labels = ["cascade delay O", "cascade delay X"]
sizes = [next_delay_cnt, delay_cnt - next_delay_cnt]
colors = ["#FF9999", "#66B2FF"]

# 파이 차트 그리기
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, wedgeprops={"edgecolor": "black"})
plt.title("cascade delay")


'''
결론: 연쇄지연 발생으로인해 항공편이 적음에도 저녁시간대에 비행기 지연이 자주 발생된다.
아침 시간대에는 연쇄 지연의 영향이 적어서 항공편이 많음에도 지연이 적게 발생한다.
'''