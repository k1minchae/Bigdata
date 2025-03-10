# Pivot table 실습 (팔머 펭귄 데이터)

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
penguins = load_penguins()

'''
문제 1: 펭귄 종별 평균 부리 길이 구하기

펭귄 데이터에서 각 종(species)별로 
평균 부리 길이(bill_length_mm)를 구하는 pivot_table()을 작성하세요.

'''
penguins.pivot_table(index="species",
                     values="bill_length_mm").reset_index()



'''
문제 2: 섬별 몸무게 중앙값 구하기

펭귄 데이터에서 각 섬(island)별로 몸무게(body_mass_g)의 
중앙값(median)을 구하는 pivot_table()을 작성하세요.

'''
penguins.pivot_table(index="island",
                    values="body_mass_g",
                    aggfunc="median").reset_index()



'''
문제 3: 성별에 따른 부리 길이와 몸무게 평균 구하기

펭귄 데이터에서 성별(sex)과 종(species)별로 
부리 길이(bill_length_mm)와 몸무게(body_mass_g)의 평균을 구하는 pivot_table()을 작성하세요.

'''
penguins.pivot_table(index=["sex", "species"],
                     values=["bill_length_mm", "body_mass_g"]).reset_index()



'''
문제 4: 종과 섬에 따른 평균 지느러미 길이 구하기

펭귄 데이터에서 각 종(species)과 섬(island)별로 
지느러미 길이(flipper_length_mm)의 평균을 구하는 pivot_table()을 작성하세요.

'''
pv = penguins.pivot_table(index="island",
                     values="flipper_length_mm",
                     columns="species",
                     fill_value="개체수 없음")
pv.columns.name = None
pv


'''
문제 5: 종과 성별에 따른 부리 깊이 합계 구하기

펭귄 데이터에서 종(species)과 성별(sex)별로 
부리 깊이(bill_depth_mm)의 총합(sum)을 구하는 pivot_table()을 작성하세요.

'''
penguins.pivot_table(index=["species", "sex"],
                     values="bill_depth_mm",
                     aggfunc="sum").reset_index()



'''
문제 6: 종별 몸무게의 변동 범위(Range) 구하기

펭귄 데이터에서 각 종(species)별로 몸무게(body_mass_g)의 
변동 범위 (최댓값 - 최솟값) 를 구하는 pivot_table()을 작성하세요.

💡 힌트: aggfunc에 사용자 정의 함수를 활용하세요.

'''
def max_diff_min(arr):
    return np.max(arr) - np.min(arr)

penguins.pivot_table(index="species",
                     values="body_mass_g",
                     aggfunc=max_diff_min).reset_index()