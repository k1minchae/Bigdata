import pandas as pd
import numpy as np
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

data = {
    '주소': ['서울특별시 강남구! 테헤란로 123', '부산광역시 해운대구 @센텀중앙로 45']
}
df = pd.DataFrame(data)
print(df.head(2))


# str.extract() : 정규표현식으로 매칭되는 패턴 중 첫 번째 값을 추출
df['도시'] = df['주소'].str.extract(r'([가-힣]+광역시|[가-힣]+특별시)', expand=False)
print(df.head(2))


# 모든 특수 문자 추출
special_chars = df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
print(special_chars)

'''
^ 대괄호 내에서 사용되면부정을 의미하며 해당 패턴에 포함되지 않는 문자를 의미함
a-z 소문자 알파벳a부터z까지
A-Z 대문자 알파벳A부터Z까지
0-9 숫자0부터9까지
가-힣 한글 음절을 나타내는 문자 범위
\s 공백 문자스페이스 탭 개행 문자 등

'''

df['주소'].str.extractall(r'([^a-zA-Z0-9가-힣\s])')
df['주소_특수문자제거'] = df['주소'].str.replace(r'[^a-zA-Z0-9가-힣\s]', '', regex=True)
print(df.head(2))

# r'\n'은 줄바꿈 문자가 아닌 두 개의 문자\와n을 나타냄


# 숫자만 꺼내오려면?
df['주소'].str.extractall(r'([0-9]+)')



'''
정규 표현식 학습

'''

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
