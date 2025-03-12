'''
문자열 데이터 처리 실습

'''

import pandas as pd
nycflights = pd.read_csv("../practice-data/nycflights.csv")
nycflights.info()

'''
데이터 설명

0 년
1 월
2 일
3 출발 시간
4 출발 지연 시간
5 도착 시간
6 도착 지연 시간
7 항공사
8 항공기 번호
9 항공편 번호
10 출발지
11 도착지
12 비행 시간
13 비행 거리
14 시간
15 분

'''
nycflights.head()
# 분석 주제 정할 것
# 인사이트 도출 2가지 이상
# 해당하는 인사이트를 보여주는 근거 데이터 생성

# 1. 월에 따라 지연 시간이 달라지는지
flights=pd.read_csv('./data_zip/nycflights.csv')
flights.info()
flights.head(3)

# 분석 주제 정하기
# 인사이트 도출 2가지 이상
# 해당하는 인사이트 보여주는 근거 데이터 생성


## 가장 정시출발률/정시도착률이 높은 월은? ##
# 통상적으로 +- 15분 이내가 정시 판단 기준
(flights['arr_delay']==0).sum()

# 정시 출발 여부 (dep_delay가 -15 ~ 15 사이일 때 True, 그 외는 False)
flights['on_time_dep'] = flights['dep_delay'].between(-15, 15, inclusive='both')

# 월별 정시 출발 비율 계산
monthly_on_time_dep = flights.groupby('month')['on_time_dep'].mean()
monthly_on_time_dep = (monthly_on_time_dep * 100).round(1)


# 정시 도착 여부 (arr_delay가 -15 ~ 15 사이일 때 True, 그 외는 False)
flights['on_time_arr'] = flights['arr_delay'].between(-15, 15, inclusive='both')

# 월별 정시 도착 비율 계산
monthly_on_time_arr = flights.groupby('month')['on_time_arr'].mean()
monthly_on_time_arr = (monthly_on_time_arr * 100).round(1)

# 비율을 퍼센트로 표시
monthly_on_time_arr = monthly_on_time_arr.astype(str) + '%'
monthly_on_time_dep = monthly_on_time_dep.astype(str) + '%'

# 테이블 생성
result1 = pd.DataFrame({
    '월별 정시 출발 비율': monthly_on_time_dep,
    '월별 정시 도착 비율': monthly_on_time_arr
})



## 가장 정시도착률이 높은 항공사는? ##

# 'carrier' 컬럼의 값을 항공사 이름으로 대체
airline = {
    '9E': 'Endeavor Air Inc.',
    'AA': 'American Airlines Inc.',
    'AS': 'Alaska Airlines Inc.',
    'B6': 'JetBlue Airways',
    'DL': 'Delta Air Lines Inc.',
    'EV': 'ExpressJet Airlines Inc.',
    'F9': 'Frontier Airlines Inc.',
    'FL': 'AirTran Airways Corporation',
    'HA': 'Hawaiian Airlines Inc.',
    'MQ': 'Envoy Air',
    'OO': 'SkyWest Airlines Inc.',
    'UA': 'United Air Lines Inc.',
    'US': 'US Airways Inc.',
    'VX': 'Virgin America',
    'WN': 'Southwest Airlines Co.',
    'YV': 'Mesa Airlines Inc.'
}
flights['airline'] = flights['carrier'].replace(airline)


# 정시 출발 여부 (dep_delay가 -15 ~ 15 사이일 때 True, 그 외는 False)
flights['on_time_dep'] = flights['dep_delay'].between(-15, 15, inclusive='both')
# 항공사별 정시 출발 비율 계산
airline_on_time_dep = flights.groupby('airline')['on_time_dep'].mean()
airline_on_time_dep = (airline_on_time_dep * 100).round(1)

# 정시 도착 여부 (arr_delay가 -15 ~ 15 사이일 때 True, 그 외는 False)
flights['on_time_arr'] = flights['arr_delay'].between(-15, 15, inclusive='both')
# 항공사별 정시 도착 비율 계산
airline_on_time_arr = flights.groupby('airline')['on_time_arr'].mean()
airline_on_time_arr = (airline_on_time_arr * 100).round(1)

# 비율을 퍼센트로 표시
airline_on_time_arr = airline_on_time_arr.astype(str) + '%'
airline_on_time_dep = airline_on_time_dep.astype(str) + '%'


# 테이블 생성
result2 = pd.DataFrame({
    '항공사별 정시 출발 비율': airline_on_time_dep,
    '항공사별 정시 도착 비율': airline_on_time_arr
})

# 월별 정시도착률이 가장 높은, 가장 낮은 항공사와 그 비율은?
# 월별로 항공사별 정시 도착률 계산
monthly_airline_on_time_arr = flights.groupby(['month', 'airline'])['on_time_arr'].mean() * 100

# 월별 최대, 최소값 찾기
monthly_max_airline = monthly_airline_on_time_arr.groupby('month').idxmax() 
monthly_min_airline = monthly_airline_on_time_arr.groupby('month').idxmin()  

# 월별 최대, 최소값에 대한 정시 도착률을 추출
monthly_max_value = monthly_airline_on_time_arr[monthly_max_airline]
monthly_min_value = monthly_airline_on_time_arr[monthly_min_airline]




# 2. 시간대 별 항공편 수 분석
# 시간대별로 지연 시간이 얼마나 달라지는지?
def divide_hour(hour):
    if 6 <= hour < 12:
        return '아침'
    if 12 <= hour < 18:
        return '점심'
    if 18 <= hour < 24:
        return '저녁'
    return '새벽'

nycflights['time_of_day'] = nycflights['hour'].apply(divide_hour)

# 2-1) 공항별, 시간대 별로 항공편수가 몇개있는지
nycflights.groupby(['origin', 'time_of_day']).size()
# 뉴어크 리버티 국제공항
# EWR     새벽              264
#         아침             4452
#         점심             4409
#         저녁             2646

# 존 F. 케네디 국제공항
# JFK     새벽              254
#         아침             3794
#         점심             3906
#         저녁             2943

# 라과디아 공항
# LGA     새벽              338
#         아침             3909
#         점심             3754
#         저녁             2066
'''
항공편 수
아침/점심 >>> 저녁 >>>>>>>>>>> 새벽
'''

# 15분 이상 지연된 비행기들
delayed_flights = nycflights.loc[nycflights['dep_delay'] >= 15, :]

# 지연된 비행기를 출발 공항별, 시간대별로 분류
delayed_flights.groupby(['origin', 'time_of_day']).size()
# origin  time_of_day
# EWR     새벽               25
#         아침              675
#         점심             1422
#         저녁             1419

# JFK     새벽               78
#         아침              461
#         점심              992
#         저녁             1215

# LGA     새벽               15
#         아침              404
#         점심              967
#         저녁              892
'''
- 항공편 수 대비 지연 비율이 가장 높은 시간대는 점심과 저녁.
- 항공편을 예약할 때 아침 시간대를 선택하는 것이 상대적으로 지연 가능성이 낮음.
- 점심 및 저녁 시간대는 지연 가능성이 크므로 주의 필요.

- 항공편 수가 많은 아침에 가장 많은 지연을 예상했으나 
  아침보다 점심이, 그리고 저녁이 확연히 지연비율이 높음.
  
- Q) 앞에 항공편이 지연되는 것이 뒷 항공편에 영향을 미쳐서 
     항공편이 적음에도 저녁시간에 많은 지연이 발생되는것이 아닐까?
'''

import numpy as np

# 연쇄지연 여부 분석A
# 출발 시간 기준으로 정렬
sorted_flight = nycflights.sort_values(['year','month', 'day', 'dep_time'], ascending=True)

# 같은 날, 이전 시간에 출발한 항공편의 도착 지연 정보 추가
sorted_flight['prev_arr_delay'] = sorted_flight.groupby(['year', 'month', 'day'])['arr_delay'].shift(1)
sorted_flight['prev_arr_delay']

# 연쇄 지연 여부 분석 (이전 항공편의 도착 지연이 현재 항공편의 출발 지연에 영향을 주었는지)
delay_cnt = len(sorted_flight.loc[sorted_flight['dep_delay'] >= 15, :]) # 7305
next_delay_cnt = len(sorted_flight.loc[(sorted_flight['dep_delay'] >= 15) & (sorted_flight['prev_arr_delay'] >= 15), :]) # 2896
print(f"연쇄 지연을 겪는 비율: {next_delay_cnt / delay_cnt * 100:.2f}%") # 지연된 비행기중 약 40%가 연쇄 지연을 겪고 있음.

'''
결론: 연쇄지연 발생으로인해 항공편이 적음에도 저녁시간대에 비행기 지연이 자주 발생된다.
아침 시간대에는 연쇄 지연의 영향이 적어서 항공편이 많음에도 지연이 적게 발생한다.
'''


# 비행기 출/도착 데이터 분석
# 분석 주제 정하기
# 인사이트 도출 2가지 이상 
# 해당하는 인사이트를 보여주는 근거 데이터 생성 

nycflights
# 비행 거리에 따른 지연 시간(출발 지연, 도착 지연)
# dep_delay 출발 지연 / arr_delay 도착 지연 / distance 비행거리 

g= nycflights.loc[:,['dep_delay','distance']]
g
# 가설 : 비행 거리가 평균 비행거리보다 길수록 출발 지연 시간이 평균 출반 지연시간보다 커질 것이다. 
# 비교를 위해 표준화를 시킴
def standardize(x):
    return(x-np.mean(x))/np.std(x)
r=g.select_dtypes('number').apply(standardize)
r
# 표준화는 각 변수의 평균이 0, 표준편차가 1이 되도록 변환하는 작업입니다.

# 출발지연시간 표준화 → 출발지연시간의 평균과 표준편차를 기준으로 변환
# 비행시간 표준화 → 비행시간의 평균과 표준편차를 기준으로 변환

#표준화된 값이 0에 가까울수록 평균에 가까운 값
#표준화된 값이 양수면 평균보다 크고, 음수면 평균보다 작음

correlation = r.corr()
correlation
#상관계수는 두 변수 사이의 상관관계(관련성)를 숫자로 나타낸 값
#두 값이 어떻게 함께 변하는지를 측정합니다.
#값의 범위는 -1 ~ 1 사이입니다.
#+1	출발지연시간이 길수록 거리도 길어짐
#-1	출발지연시간이 길수록 거리가 짧아짐
#0	상관관계 없음	출발지연시간과 거리가 서로 관계 없음


# 출발 지연시간과 비행거리 간에는 상관관계가 없다. 상관계수가 0에 가깝기 때문.

# 비행시간과 도착 지연시간 간의 관계 
a= nycflights.loc[:,['arr_delay','distance']]
def standardize(x):
    return(x-np.mean(x))/np.std(x)
b=a.select_dtypes('number').apply(standardize)
b

correlation= b.corr()
correlation

# 도착 지연시간과 비행거리 간에도 상관관계가 없다...