import pandas as pd
import numpy as np

df = pd.read_csv('../practice-data/dat.csv')
df.head()
df.info()

'''
1. school 학생이 소속된 학교 ('GP' Gabriel Pereira 또는 'MS' Mousinhoda Silveira)
2. sex 학생의 성별 'F'여성 또는 'M'남성
3. paid 추가 과외 수업 여부 'yes' 'no'
4. famrel 가족 관계의 질 (1매우 나쁨, 2 나쁨, 3 보통, 4 좋음, 5 매우 좋음)
5. freetime 자유 시간의 질 (1매우 적음, 2 적음, 3 보통, 4 많음, 5 매우 많음)
6. goout 친구와 외출 빈도 (1매우 적음, 2 적음, 3 보통, 4 많음, 5 매우 많음)
7. Dalc 주중 음주량 (1매우 적음, 2 적음, 3 보통, 4  많음, 5 매우 많음)
8. Walc 주말 음주량 (1매우 적음, 2 적음, 3 보통, 4  많음, 5 매우 많음)
9. health 현재 건강 상태 (1매우 나쁨, 2 나쁨, 3 보통, 4 좋음, 5 매우 좋음)
10. absences 결석 일수
11. grade 최종 성적

'''

# grade: 0 ~ 11
set(df['grade'])

# 여러 칼럼 선택
df.loc[:, ['school', 'sex', 'paid', 'goout']]

# 이름 변경: rename()
df = df.rename(columns={'Dalc': 'dalc', 'Walc': 'walc'})

# 데이터 타입 변경: astype()
# 결측치 존재
pd.isna(df).sum()   # goout 에만 10개의 결측치 존재
df.loc[df.loc[:, 'goout'].isna() == True, :]
'''
Nan 값 처리하는 법
1. 최빈값 대체
- 기존의 데이터 형태인 정수 값으로 대체 가능
- 전체 변수의 평균 값이 변경될 수 있음
2. 평균값 대체
- 전체 변수의 평균값이 그대로 유지
- 데이터에 극단값이 존재할 경우 영향을 받을 수 있음
- 소수점으로 인해 데이터 타입이 변할 수 있음
'''

df.astype({'goout': 'Int64'})

# nan 값 최빈값 대체
most_val = df.loc[:, 'goout'].mode()[0]    # 최빈값
is_na = df.loc[:, 'goout'].isna() == True   # 필터링조건

df.loc[is_na, 'goout'] = most_val   # 최빈값 대체
# df['goout'].fillna(most_val)

# 타입변경: astype
df = df.astype({'goout': 'int64'})

# assign() 새로운 파생 변수 생성
def classify_famrel(famrel):
    if famrel <= 2:
        return 'Low'
    if famrel <= 4:
        return 'Medium'
    return 'High'

df = df.assign(famrel_quality=df['famrel'].apply(classify_famrel))
df = df.rename(columns={'famrel_quality': 'farmel_q'})

# 데이터 타입으로 칼럼 선택하는 방법
df.select_dtypes('number')
df.select_dtypes('object')

# 데이터 타입으로 선택한 데이터프레임에 apply 적용
def standardize(x):
    return (x - np.nanmean(x))/ np.std(x)
std = df.select_dtypes('number').apply(standardize)
std



# 1. 사교육 정말 해야 하는가?
# 사교육을 받는 학생만 출력
df.loc[df['paid'] == 'yes', 'grade'].describe()
df.loc[df['paid'] == 'no', 'grade'].describe()


# 2. 음주량이 건강 상태에 미치는 효과
# 3. 4~9번 변수들이 성적에 얼마나 영향을 미치는가?
# 꼭, 특정 변수를 분석 대상(종속 변수)으로 설정할 필요는 없다.


'''
연습문제: 학교 성적 데이터

'''

# df 데이터 프레임의 정보를 출력하고 각 열의 데이터 타입을 확인하세요
df = pd.read_csv("../practice-data/grade.csv")
df.info()

# midterm 점수가85점 이상인 학생들의 데이터를 필터링하여 출력하세요
df.loc[df['midterm'] >= 85, :]

# final 점수를 기준으로 데이터 프레임을 내림차순으로 정렬하고 
# 정렬된 데이터 프레임의 첫 5행을 출력하세요
df.sort_values('final', ascending=False).head()

# gender 열을 기준으로 데이터 프레임을 그룹화하고
# 각 그룹별 midterm과 final의 평균을 계산하여 출력하세요
df.groupby('gender')['midterm'].mean()

# student_id 열을 문자열 타입으로 변환하고 
# 변환된 데이터 프레임의 정보를 출력하세요
df['student_id'] = df['student_id'].astype('str')
df.info()

# assignment 점수의 최대값과 최소값을 가지는 행을 각각 출력하세요
max_idx = np.where(df['assignment'] == df['assignment'].max())[0]
min_idx = np.where(df['assignment'] == df['assignment'].min())[0]
df.iloc[max_idx, :]
df.iloc[min_idx, :]

# midterm final assignment 점수의 평균을 계산하여 
# average 열을 추가하고 첫5행을 출력하세요
df.loc[:, 'average'] = np.nanmean(df.loc[:, ['midterm', 'final', 'assignment']], axis=1)
df.head()

# 아래의 추가 데이터를 생성하고 기존 데이터 프레임과 
# student_id를 기준으로 병합하여 출력하세요
additional_data = {
    'student_id': ['1', '3', '5', '7', '9'],
    'club': ['Art', 'Science', 'Math', 'Music', 'Drama']
}
df_additional = pd.DataFrame(additional_data)
pd.merge(df, df_additional, how='left', on='student_id')

# gender를 인덱스로 student_id를 열로 사용하여 
# average 점수에 대한 피벗 테이블을 생성하고 출력하세요
pivot_table = df.pivot_table(values='average', index='gender', columns='student_id')
pivot_table


# midterm final assignment의 평균을 구하고 average 열을 생성하시오 
# 또한 성별 성적 유형 assignment average final midterm별 평균 점수를 계산하시오
df.groupby(['gender']).mean(numeric_only=True)

# midterm final assignment의 평균을 구하고 average 열을 생성하시오 
# 또한 최대 평균 성적을 가진 학생의 이름과 평균 성적을 출력하시오
max_avg = df['average'].max()
df.loc[df['average'] == max_avg, ['name', 'average']]

