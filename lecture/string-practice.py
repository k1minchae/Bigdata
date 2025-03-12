# 정규 표현식 연습

import pandas as pd
import numpy as np

df = pd.read_csv('../string-data/regex_practice_data.csv')

# 이메일 주소 찾기
# \w : 단어 문자알파벳 숫자 밑줄에 매칭
df['전체_문자열'].str.extract(r'([\w\.]+@+[a-z\.]+)')

# 휴대폰 번호 찾기
df['전체_문자열'].str.extract(r'(010-[0-9\-]+)').dropna()

# 일반 전화번호 (지역번호 포함) 찾기
numbers = df['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
not_phone = ~numbers.iloc[:, 0].str.startswith('010')

nums = df.loc[not_phone, :]['전체_문자열'].str.extract(r'(\d+-[0-9\-]+)')
nums

# 4. 주소에서 '구' 단위만 추출하기
df['전체_문자열'].str.extract(r'([가-힣]+구+\b)')

# 5. 날짜(YYYY-MM-DD) 형식 찾기
df['전체_문자열'].str.extract(r'(\d{4}-[\d\d]-[\d]+)').dropna()

# 6. 모든 날짜 형식 찾기 (YYYY-MM-DD, YYYY/MM/DD, YYYY.MM.DD 포함)
df['전체_문자열'].str.extract(r'(\d{4}[-/.]\d{2}[-/.]\d{2}+)')

# 7. 가격 정보(₩ 포함) 찾기
df['전체_문자열'].str.extract(r'(₩[0-9,]+)')

# 8. 가격에서 숫자만 추출하기 (₩ 제거)
df['전체_문자열'].str.extract(r'(₩[0-9,]+)').iloc[:, 0].str.replace('₩', '')
df['전체_문자열'].str.extract(r'₩([0-9,]+)')

# 9. 이메일의 도메인 추출하기
df['전체_문자열'].str.extract(r'(@[a-z\.]+)').iloc[:, 0].str.replace('@', '')
df['전체_문자열'].str.extract(r'@([a-z\.]+)')

# 10. 이름만 추출하기
df['전체_문자열'].str.extract(r'([가-힣]+)')
