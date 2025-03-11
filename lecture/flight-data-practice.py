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
8 꼬리 번호
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

# 1. 월에 따라 지연 시간이 달라지는지 -> 규진

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
비행기 수요
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

# 연쇄지연 여부 분석A
# 출발 시간 기준으로 정렬
sorted_flight = nycflights.sort_values(['year','month', 'day', 'dep_time'], ascending=True)
sorted_flight = sorted_flight.fillna(0)

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

# 3. 비행 거리에 따른 지연 시간 -> 보경

# 4. 공항별, 노선별 ,...