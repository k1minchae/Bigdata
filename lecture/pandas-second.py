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

avg = (mydata['total'] / 2).rename('average')   # avg 시리즈를 만듦
mydata = pd.concat([mydata, avg], axis=1)       # mydata 에 avg Series 추가해서 새로운 df 로 만듦

# 열 삭제
del mydata['gender'] 

mydata = mydata.iloc[:, [0, 1, 2, 4, 3]]    # 열 순서 바꾸기