import pandas as pd
bike = pd.read_csv('../practice-data/bike_data.csv')
bike.info()

# 데이트타임형으로 변환
# 범주형 변수와 순서형 변수는 인코딩 방법이 다름
bike = bike.astype({'datetime': 'datetime64[ns]'})

# 계절 season == 1일 때 가장 대여량이 많은 시간대 hour을 구하시오
bike['hour'] = bike['datetime'].dt.hour
max_lent_hour = bike.loc[bike['season'] == 1, ['count', 'hour']].groupby('hour').sum().idxmax()

# 각 계절season별 평균 대여량count을 구하시오
bike.loc[:, ['season', 'count']].groupby('season').mean()

# 특정 달month 동안의 총 대여량count을 구하시오
bike['month'] = bike['datetime'].dt.month
months = bike.loc[:, ['month', 'count']].groupby('month').sum()

# 가장 대여량이 많은 날짜를 구하시오
bike['date'] = bike['datetime'].dt.date
date = bike.loc[:, ['count', 'day']].groupby('day').sum().idxmax()
val = bike.loc[:, ['count', 'day']].groupby('day').sum().max()

# 시간대hour별 평균 대여량count을 구하시오
bike.loc[:, ['hour', 'count']].groupby('hour').mean()

# 특정 요일weekday 동안의 총 대여량count을 구하시오
bike['weekday'] = bike['datetime'].dt.weekday
label = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
def get_weekday(num):
    return label[num]
bike['weekday'] = bike['weekday'].apply(get_weekday)
weekdays = bike.loc[:, ['weekday', 'count']].groupby('weekday').sum()


# 주어진 데이터를 사용하여 넓은 형식에서 긴 형식으로 변환하시오.
# casual과 registered열을 하나의 열로 변환하고 각 기록의 대여 유형과 
# 대여수를 포함하는 긴 형식 데이터프레임을 만드시오
bike = pd.read_csv('../practice-data/bike_data.csv')
bike.info()
bike.head()
bike_melted = bike.melt(
    id_vars=['datetime', 'season'],  # 유지할 열들
    var_name='user_type',       # 새로 생성될 열 이름
    value_vars=['casual', 'registered'], # 녹일 열
    value_name='user_count' # 새로 생성될 값 열의 이름
)

# 이전에 생성한 긴 형식 데이터프레임을 활용하여 각 계절별로 casual과 registered 
# 사용자의 평균 대여 수 count를 구하시오
bike_melted.groupby(['season', 'user_type'])['user_count'].mean().reset_index()