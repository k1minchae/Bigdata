import pandas as pd
import numpy as np

# 데이터 프레임 생성 (key 값: column)
df = pd.DataFrame({
    # value 값 안에 있는 원소의 개수가 같아야 한다.
    # type 은 달라도 된다.
    'col1': ['one', 'two', 'three', 'four', 5],
    'col2': [6, 7, 8, 9, 10]
})

df['col1']  # 시리즈 (1차원)
df.col1 # 이렇게 가져올 수도 있다.
df.columns # 칼럼들의 이름을 전부 출력
df['col1'].dtype    # object
df['col2'].dtype    # int64

print(df.shape)   # (5, 2)

'''
각 컬럼에 접근하면 시리즈이고, 전체에 접근하면 데이터프레임
'''

data = [10, 20, 30]
df_s = pd.Series(data)
print(df_s)
df_s.shape  # (3,)

df_s = pd.Series(data, index=['a', 'b', 'c'])  # index 지정
df_s['a']   # 10
df_s[0] # 경고

# 빈 데이터 프레임 만들기
my_df = pd.DataFrame({
    '실수변수': pd.Series(dtype='float'),
    '정수변수': pd.Series(dtype='int'),
    '범주형변수': pd.Series(dtype='category'),
    '논리변수': pd.Series(dtype='bool'),
    '문자열변수': pd.Series(dtype='str')
})
print(my_df.dtypes)


# 데이터 채우면서 만들기
my_df = pd.DataFrame({
    'name': ['issac', 'bomi'],
    'birthmonth': [5, 4]
})
print(my_df)
print(my_df.dtypes)

# 시험 성적에 관한 DataFrame 만들기
exam = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "gender": ['F', 'M', 'F', 'M', 'M'],
    "midterm": [38, 42, 53, 48, 46],
    "final": [46, 67, 56, 54, 39]
}, index=['first', 'second', 'third', 'fourth', 'fifth'])

# gender 만 시리즈로 만들고 싶을 때
pd.Series(exam["gender"])

# 시리즈에 컬럼 명을 넣을 수 있다.
my_s = pd.Series(['F', 'M', 'F', 'M', 'M'], name="gender", index=exam.index)
midterm_series = pd.Series(exam["midterm"], name="midterm", index=exam.index)

df = pd.concat([my_s, midterm_series], axis=1)  # 각각의 시리즈를 합칠 수 있다.

# 그대신 index 값이 다르면 NaN 이들어감 (합집합으로 구현되어있음)
# gender	midterm
# 0	F	NaN
# 1	M	NaN
# 2	F	NaN
# 3	M	NaN
# 4	M	NaN
# first	NaN	38.0
# second	NaN	42.0
# third	NaN	53.0
# fourth	NaN	48.0
# fifth	NaN	46.0

# index 가 같아야 합칠 수 있음
df = pd.DataFrame({
    "gender__": my_s,
    "midterm": midterm_series
})


# 외부 데이터 가져오기
url = "https://bit.ly/examscore-csv"
mydata = pd.read_csv(url)
mydata.shape  # (30, 4)
print(mydata.head())    # 앞에 n줄만 (기본값: 5) 보여주는 메서드


df.iloc[0]  # 0번째 row를 가져옴

# 여러 데이터를 동시에 필터링 (column)
mydata[['gender', 'student_id']]

# TypeError : numpy 처럼 행과 열을 선택할 수 없다. => iloc 사용해야함
mydata[['gender', 'student_id'], :]


'''
numpy 처럼 슬라이싱 하는법: iloc, loc
'''
# iloc: 위치 기반 인덱싱 가능
mydata.iloc[:, 0:2] # 모든 행, 0 ~ 1열
mydata.iloc[1:4, 1:3]

# loc: 열에 라벨 기반 인덱싱 가능
mydata.loc[:, ['gender', 'student_id']]
mydata.loc[1:5, ['student_id']]

# iloc은 위치 기반으로 하기 때문에 index명과는 관계없다.
mydata2 = pd.DataFrame(mydata.iloc[:4, :3])
mydata2.index = ['a', 'b', 'c', 'd']
mydata2.iloc[0:2, 0:2]
# 	    student_id	gender
# a	    1	        F
# b	    2	        M


# ★ iloc 함수는 결과값의 타입이 변동함
mydata.iloc[0, 0]       # 열 1개, 행 1개일 때는 Data 1개가 나옴 np.int64(1)
mydata.iloc[0, 1]       # str('F')
mydata.iloc[0:2, 2]     # 열 1개: Series
mydata.iloc[[0, 2], 2]  # 행 리스트 형식: Series
mydata.iloc[0:2, [2]]   # 열 리스트 형식: DataFrame
mydata.iloc[1, 0:2]     # 행 1개: Series
mydata.iloc[0:2, 0:2]   # 열/행 여러개: DataFrame
mydata2.iloc[:, [0]].squeeze()  # Series로 짜내기
mydata2.iloc[:, [0, 2]]   # 연속되지않은 열들 행들을 리스트 형태로 받기
mydata2.iloc[[0, 2], :] # DataFrame


mydata.shape  # (30, 4)

# 짝수번째 행들만 선택하려면?
mydata.iloc[1::2, :]
mydata.iloc[np.arange(1, 30, 2), :]

a = np.array([1, 2, 3])
mydata.iloc[a, :]  # a에 index가 있는 row들을 DataFrame으로 변환


# ★ 필터링
mydata['gender'][0] # 'F'
np.array(mydata['gender'])  # array(['F', 'M', 'F',..., 'M', 'F', 'M'], dtype='object')
check_f = np.array(mydata['gender']) == 'F'  # array([ True, False,  True,...], dtype=bool)

# numpy 벡터를 넣으면 iloc으로 필터링 가능
mydata.iloc[check_f]

# series(bool)을 넣으면 loc으로 필터링 가능
mydata.loc[mydata['gender'] == 'F']

