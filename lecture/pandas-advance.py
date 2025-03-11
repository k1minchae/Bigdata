import pandas as pd
import numpy as np

# 외부 데이터 가져오기
url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)

mydata[mydata['gender'] == 'F', :]      # 에러 발생
mydata.loc[mydata['gender'] == 'F', :]  # 작동O: DataFrame
mydata[mydata['gender'] == 'F']         # 작동O: DataFrame

check = np.array(mydata['gender'] == 'F')
mydata.iloc[check, :]                   # 작동O: DataFrame

# 조건 필터링
mydata[mydata['midterm'] <= 15]

# Q. 중간고사 점수 45~60점 사이 학생은 몇명인가요?
condition = (mydata['midterm'] >= 45) & (mydata['midterm'] <= 60)
mydata[condition].shape[0]  # 10명

'''
숫자를 사용할 땐 iloc
label 을 사용할 땐 loc 을 사용하는 것이 좋다.
'''


# isin([]): 특정 값이 존재하는지 여부
# loc과 함께 사용하면 한번에 필터링 가능

mydata['midterm'].isin([28, 38, 52])    # 시리즈가 제공하는 메서드
# 0      True
# 1     False
# 2     False
# 3     False
# 4     False
# 5     False
#        ...    => boolean Series

# 범주형 변수를 필터링 하기 좋다.
mydata.loc[mydata['midterm'].isin([28]), :]

# Boolean Series 앞에 ~를 붙이면 False 인 것만 필터링
mydata.loc[~mydata['midterm'].isin([28, 38, 52]), :]


# 데이터에 빈칸이 뚫려있는 경우
mydata.iloc[3, 2] = np.nan
mydata.iloc[10, 3] = np.nan
mydata.iloc[13, 1] = np.nan

mydata['gender'].isna() # gender 에 NaN 이 있는지 boolean Series
mydata.isna()       # 전체 DataFrame 에 NaN 이 있는지 boolean DataFrame

mydata.loc[mydata['gender'].isna(), :]

# mydata 에서 중간고사와 기말고사가 다 채워진행들을 가져오세요
mydata.loc[~(mydata['midterm'].isna()) & ~(mydata['final'].isna()), :]

mydata['midterm'].isna().sum() # Na 체크
mydata.loc[mydata['midterm'].isna(), 'midterm'] = 50
mydata.loc[mydata['final'].isna(), 'final'] = 30

mydata['total'] = mydata['midterm'] + mydata['final']
mydata.dropna()
mydata

# pd.concat([], axis=) 데이터 프레임이나 시리즈를 합치는 함수
avg = (mydata['total'] / 2).rename('average')   # avg 시리즈를 만듦
mydata = pd.concat([mydata, avg], axis=1)       # mydata 에 avg Series 추가해서 새로운 df 로 만듦

# 열 삭제
del mydata['gender'] 

mydata = mydata.iloc[:, [0, 1, 2, 4, 3]]    # 열 순서 바꾸기

df1 = pd.DataFrame({
    'A': ['A0', 'A1', 'A2'],
    'B': ['B0', 'B1', 'B2']
})

df2 = pd.DataFrame({
    'A': ['A3', 'A4', 'A5'],
    'B': ['B3', 'B4', 'B5']
})
df3 = pd.DataFrame({
    'C': ['C0', 'C1', 'C2'],
    'D': ['D0', 'D1', 'D2']
})
result = pd.concat([df1, df2])  # df1 행 아래에 df2 의 행이 추가됨 (index 중복)
result = pd.concat([df1, df2], ignore_index=True)   # index 새로 초기화
result = pd.concat([df1, df3], axis=1)


# join 속성 활용
df4 = pd.DataFrame({
    'A': ['A2', 'A3', 'A4'],
    'B': ['B2', 'B3', 'B4'],
    'C': ['C2', 'C3', 'C4']
})
# inner join: 공통 열만 합침
pd.concat([df1, df4], join='inner', ignore_index=True)

# (기본값) outer join: 모든 열 합침
pd.concat([df1, df4], join='outer', ignore_index=True)


# keys 속성 사용
# 데이터 출처를 기록하는 법 
# 각 프레임의 원본 출처를 식별하는 멀티 인덱스 생성
# 키 개수가 안맞으면 warning 발생 (단, keys 설정된것만 됨)
result = pd.concat([df1, df4], keys=['df1', 'df2'])
result.loc['df1']   # 새로운 key 로 활용가능
result.loc['df2'].iloc[1:3]


'''
데이터 프레임 메서드 정리
'''
mydata.head()       # DataFrame 상위 5개 row
mydata.tail()       # DataFrame 하위 5개 row
mydata.info()       # DataFrame 정보 (데이터 타입, row, column, non-null count, 메모리)
mydata.describe()   # DataFrame 통계요약 (count, mean, std, min, 25%, 50%, 75%, max)

mydata.sort_values(by=['midterm','final'], ascending=False)   # 정렬
mydata.mean(numeric_only=True)       # DataFrame column별 mean (시리즈로 반환)
# numeric_only 를 안해주면 문자열있을때 오류남

'''
apply 함수 이해하기

'''
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)
df.apply(max, axis=0)   # 열
df.apply(max, axis=1)   # 행

# 사용자 정의 함수를 apply 에 적용하기
def my_func(x, const=3):
    return max(x) ** 2 + const

df.apply(my_func, axis=1, const=5)    # 매개 변수를 속성에 추가하면됨
df.apply(my_func, axis=0)


# 팔머펭귄 데이터 실습
# pip install palmerpenguins
from palmerpenguins import load_penguins
penguins = load_penguins()
penguins.info()

# 각 펭귄 종별 특징 알아내서 발표
# 펭귄 종류 확인
penguins.loc[:, 'species'].unique()
# array(['Adelie', 'Gentoo', 'Chinstrap'], dtype=object)

# 종별 평균값 구하기
species_means = penguins.groupby('species').mean(numeric_only=True)
species_means = penguins.groupby(['species', 'sex']).mean(numeric_only=True)
penguins.loc[penguins['sex'].isna(), :]

# 수치가 아닌 열은 빈도수 확인
not_num = penguins.select_dtypes(include=['object']).columns
group_by_species = penguins.groupby('species')
cnt = {col: group_by_species[col].value_counts() for col in not_num}

# 부리 길이
max_bill_length = species_means['bill_length_mm'].idxmax()  
min_bill_length = species_means['bill_length_mm'].idxmin()  
max_bill_length, min_bill_length

# 부리 깊이
max_bill_depth = species_means['bill_depth_mm'].idxmax()  
min_bill_depth = species_means['bill_depth_mm'].idxmin()  

# 무게
max_body_mass = species_means['body_mass_g'].idxmax()  
min_body_mass = species_means['body_mass_g'].idxmin()  

# 날개 길이
max_flipper_length = species_means['flipper_length_mm'].idxmax()
min_flipper_length = species_means['flipper_length_mm'].idxmin()


'''
복습

'''
pd.Series(penguins["bill_length_mm"].fillna(0), dtype="int64")  # data 타입을 int 로변경 이때 nan 값있으면안됨
penguins.head(20)   # 기본값: 5
penguins.sort_values(by=["bill_length_mm", "body_mass_g"], ascending=False) # 기본값: 오름차순
round(penguins, 0).loc[:, ["bill_length_mm", "body_mass_g"]].sort_values(by=["bill_length_mm", "body_mass_g"], ascending=[False, True])
penguins["bill_length_mm"] = round(penguins["bill_length_mm"])

max_idx = penguins["bill_length_mm"].idxmax()
max_v = penguins["bill_length_mm"].max()
penguins.iloc[[max_idx],]   # DF (콜론 안써도 돌아간다?)
# penguins.loc[, 1]   # 오류
penguins.iloc[max_idx,]   # Series


# 최대 부리 길이 (60mm)인 펭귄은 몇마리?
len(penguins.loc[penguins["bill_length_mm"] == max_v, :])


# species 열을 기준으로 그룹화하여 평균 계산
penguins.groupby(['species', 'sex']).mean(numeric_only=True)

# 섬별로 쪼개서 부리 길이의 평균 구하기
penguins.groupby('island')['bill_length_mm'].mean()

list(penguins.groupby('species'))

penguins.groupby('island')['bill_length_mm'].sum().idxmax() # Series
# Biscoe : 인덱스가 문자열


# as_index = False 옵션 활용
# 기본 인덱스 추가 => 기존 index는 따로 빼서 Column 으로 만들어줌
# 기존 인덱스가 하나의 열이 되기 때문이 sort_values 활용 가능
penguins.groupby('island', as_index=False)['bill_length_mm'].sum().idxmax() # DF
penguins.groupby('island')['bill_length_mm'].sum().idxmax() # Series

# 그룹을 두개 변수로 활용하고 싶을 땐?
penguins.groupby(['species', 'island']).mean(numeric_only=True)


# merge()
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})
pd.merge(df1, df2, on='key', how='inner')   # key열 기준으로 병합, inner join
pd.merge(df1, df2, on='key', how='outer')   # left, right 도 있음
pd.concat([df1, df2])   # concat 과의 차이점 (중복 없애지 않음 그냥 이어붙이기만함)
pd.concat([df1, df2], join='inner') # 공통된 column 만 추출


# 실습
mid = pd.DataFrame({'id': [23, 10, 5, 1], 'midterm': [40, 30, 50, 20]})
final = pd.DataFrame({'id': [23, 10, 5, 30], 'final': [45, 25, 50, 47]})
pd.merge(mid, final, on='id', how='outer')


# 실습 2
# 1) 성별, 섬별 부리 길이 평균 계산
df1 = penguins.groupby(['sex', 'island'])['bill_length_mm'].mean()  # 시리즈
# 2) 성별, 섬별 부리 깊이 평균 계산
df2 = penguins.groupby(['sex', 'island'])['bill_depth_mm'].mean()   # 시리즈
# 3) 앞에 두개의 데이터프레임을 병합해서 성별, 섬별, 부리깊이, 깊이 DF 만들기
pd.merge(df1, df2, on=['sex', 'island'], how='outer')
pd.merge(df1, df2, on=['sex', 'island'], how='outer').reset_index()


# 실습 3
penguins.loc[:, "bill_depth_mm"].mean()

# 이렇게하면 여러줄 코딩 가능
(penguins
    .loc[:, "bill_length_mm"]
    .mean())


'''
데이터 재구조화: 전처리 과정에서 중요
긴 코드를 효율적으로 줄일 수 있다.
★ 데이터를 깔끔하게 관리해야 함
- 행 하나가 관찰값 1개
'''

# melt()
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
print(df)
##   Date Temperature Humidity
## 0 2024-07-01 10 60
## 1 2024-07-02 20 65
## 2 2024-07-03 25 70
## 3 2024-07-03 20 21



# data를 wide -> long
df_melted = pd.melt(df,
    id_vars=['Date'],
    value_vars=['Temperature', 'Humidity'],
    var_name='Variable',    # 온도 습도가 Variable 컬럼으로 내려옴
    value_name='Value')
df_melted
##    Date Variable Value
## 0 2024-07-01 Temperature 10
## 1 2024-07-02 Temperature 20
## 2 2024-07-03 Temperature 25
## 3 2024-07-03 Temperature 20
## 4 2024-07-01 Humidity 60
## 5 2024-07-02 Humidity 65

wide_data = pd.DataFrame({
    'country': ['Afghanistan', 'Brazil', 'China'],
    "2024년": [745, 37737, 212258],
    "2025년": [2666, 80488, 213766]
})
wide_data

df_long = pd.melt(wide_data,
    id_vars=['country'],
    value_vars=["2024년", "2025년"],
    var_name = 'year',
    value_name='cases'
)
df_long



# pivot() : long -> wide
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03'],
    'Temperature': [10, 20, 25],
    'Humidity': [60, 65, 70]
}
df = pd.DataFrame(data)
# long 하게 변환
df_melted = pd.melt(df,
                    id_vars=['Date'],
                    value_vars=['Temperature', 'Humidity'],
                    var_name='Variable',
                    value_name='Value')
df_melted

# pivot 함수를 사용해서 다시 wide 하게 원상복귀
df_pivoted = df_melted.pivot(index='Date',
                             columns='Variable',
                             values='Value').reset_index()
df_pivoted



wide_data = pd.DataFrame({
    'country': ['Afghanistan', 'Brazil', 'China'],
    "2024": [745, 37737, 212258],
    "2025": [2666, 80488, 213766]
})
wide_data

df_long = pd.melt(wide_data,
    id_vars=['country'],
    value_vars=["2024", "2025"],
    var_name = 'year',
    value_name='cases'
)
df_long

# pivot(): wide 하게 변형할 열과 해당 value 값에 들어갈 열을 고르기
wide_2 = df_long.pivot(index='country',
              columns='year',
              values='cases').reset_index()
wide_2.columns.name = None
wide_2.shape    # (3, 3) year, country 가 들어있는 index 부분은 shape 에들어가지않음


# pivot_table()
data = {
    'Date': ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-03'],
    'Temperature': [10, 20, 25, 20],
    'Humidity': [60, 65, 70, 21]
}
df = pd.DataFrame(data)
df
df_melted2 = pd.melt(df,
        id_vars=['Date'],
        value_vars=['Temperature', 'Humidity'],
        var_name='Weather Factor',
        value_name='TorH')

# 이렇게 하면 에러 발생
# Index contains duplicate entries : 중복된 index가 있어서 오류
df_melted2.pivot(index='Date',
                 columns='Weather Factor',
                 values='TorH').reset_index()

# pivot_table() 활용
# aggfunc 옵션을 활용하여 중복이 있는 날짜의 값을 평균으로 대치
# aggfunc 집계 함수 (기본: mean, sum, count)
df_pivot_table = df_melted2.pivot_table(index='Date',
                                        columns='Weather Factor',
                                        values='TorH').reset_index()
df_pivot_table.columns.name = None
df_pivot_table


# 예제
df = pd.DataFrame({
    'school': ['A', 'A', 'B', 'B', 'C', 'C'],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'city': ['North', 'South', 'North', 'South','North', 'South'],
    'midterm': [10, 20, 30, 40, 50, 60],
    'final': [5, 15, 25, 35, 45, 55]
})
df

'''
깔끔한 데이터의 조건
1. 각 칼럼이 하나의 변수를 의미
2. 각 행이 하나의 관측치를 나타낸다.
'''
df.pivot_table(index='school',
               columns='')

# 학교별 중간고사 평균 => Group by 와 같은 효과
df.pivot_table(index='school',
                columns='city',
                values=['midterm', 'final'],
                aggfunc='mean'
                ).reset_index()


# 인덱스가 여러개
df.pivot_table(index=['school', 'gender'],
            #    columns='city',
               values=['midterm', 'final'],
               aggfunc='mean').reset_index()
# 결과값은 무조건 index 중복 X

df.pivot_table(index='school',
            #    columns='city',
               values='midterm',
               aggfunc='sum').reset_index()

df.pivot_table(index='school',
            #    columns='city',
               values='midterm',
               aggfunc='last').reset_index()    # 마지막 꺼 가져오는 집계함수

# 벡터 원소들을 더한 수 제곱을 하는 함수 my_f
def my_f(arr):
    return np.sum(arr) ** 2

df.pivot_table(index='school',
            #    columns='city',
                values='midterm',
                aggfunc=my_f).reset_index()


# f로 시작하는 칼럼 찾기
df.columns.str.startswith('f')  # 결과값: boolean np array

# c로 끝나는 칼럼 찾기
df.columns.str.endswith('c') 

# f를 포함하는 칼럼 찾기 
df.columns.str.contains('f') 


new_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [90, 85, 88]
})

# 데이터를 csv 파일로 저장하는법
new_data.to_csv("../practice-data/my-data.csv", index=False, encoding="utf-8")

# Excel 파일로 저장하는 방법
new_data.to_excel("../practice-data/my-ex-data.xlsx", sheet_name="sheet1")

# Parquet 파일로 저장하기 (대용량 데이터 저장 포맷)
new_data.to_parquet("../practice-data/my-pq-data.parquet", engine="pyarrow")

# JSON 파일로 저장하기
new_data.to_json("../practice-data/my-data.json", orient="records", indent=4)



'''
날짜와 시간 다루기

'''
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

data = {
    'date': ['2024-01-01 12:34:56', '2024-02-01 23:45:01', '2024-03-01 06:07:08'],
    'value': [100, 201, 302]
}

date_data = pd.DataFrame(data)
date_data['date'] = pd.to_datetime(date_data['date'])
date_data.dtypes

pd.to_datetime('02-2024-01', format='%m-%Y-%d')
pd.to_datetime('2025년 3월 11일', format='%Y년 %m월 %d일')
date_data['wday'] = date_data['date'].dt.day_name() # 요일 추출
date_data['wday'] = date_data['date'].dt.weekday # 월요일: 0
date_data['date'].dt.hour   # 시간 정보
date_data['date'].dt.second # 초 정보

date_data['yr'] = date_data['date'].dt.year
date_data['mn'] = date_data['date'].dt.month
date_data['day'] = date_data['date'].dt.day
date_data = date_data.rename(columns={'yr': 'year', 'mn': 'month'})

# 일정한 간격의 날짜 범위 생성
date_range = pd.date_range(start='2021-01-01', end='2022-01-10', freq='MS') # Month Start
date_range = pd.date_range(start='2021-01-01', end='2022-01-10', freq='D')  # Day
date_range = pd.date_range(start='2021-01-01', end='2022-01-10', freq='YE') # Year End
date_range


'''
문자열 다루기

'''
data = {
    '가전제품': ['냉장고', '세탁기', '전자레인지', '에어컨', '청소기'],
    '브랜드': ['LG', 'Samsung', 'Panasonic', 'Daikin', 'Dyson']
}
df = pd.DataFrame(data)
df['가전제품'].str.len()
df['브랜드'] = df['브랜드'].str.lower() # dyson
df['브랜드'] = df['브랜드'].str.upper() # DYSON
df['브랜드'] = df['브랜드'].str.title() # Dyson
'LG LG'.title()


df.columns.str.contains('l')    # np.array
df['브랜드'].str.contains('l')   # 시리즈

df['브랜드'].str.replace('a', 'aaaa')   # a를 'aaaa'로 대체
df['브랜드'].str.split('a', )   # a가 빠지고 그거기준으로 리스트형식으로 나뉨

