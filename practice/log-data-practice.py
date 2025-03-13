'''
앱 로그데이터 분석

'''
import pandas as pd

df = pd.read_csv('../practice-data/logdata.csv')
log = df['로그']

log.head()

# 로그 칼럼에서 숫자 정보만 추출하시오
log.str.replace(r'\D+', ' ', regex=True)

# 로그 칼럼에서 모든 시간 정보를 추출하시오
log.str.extract(r'(\b\d+:\d+:\d+)')

# 로그 칼럼에서 한글 정보만 추출하시오
log.str.extract(r'([가-힣]+)')

# 로그 칼럼에서 특수 문자를 제거하시오
log.str.replace(r'[^a-zA-Z0-9\s]+', '', regex=True)

# 로그 칼럼에서 유저Amount 값을 추출한 후 각 유저별Amount의 평균값을 계산하시오
log.str.extract(r'([가-힣]+)')
df['Amount'] = log.str.extract(r'Amount:\s*(\d+)').dropna()
df['User'] = log.str.extract(r'([가-힣]+)')

# 평균 계산을 위해 변환
df = df.astype({'Amount': 'float64'})
df.groupby('User', dropna=True)['Amount'].mean()