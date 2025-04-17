# 다중선형회귀파고들기 (회귀계수 인터렉션 항의 의미)
from palmerpenguins import load_penguins
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
df = load_penguins()
# 입력값: 펭귄 종, 부리 길이
# 결과값: 부리 깊이
# 선형회귀 모델 적합하기 문제
model=LinearRegression()
penguins=df.dropna()
penguins_dummies = pd.get_dummies(penguins, 
                            columns=['species'],
                            drop_first=False)
x = penguins_dummies[["bill_length_mm", "species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies['bill_depth_mm']
x["bill_Chinstrap"] = x["bill_length_mm"] * x["species_Chinstrap"]
x["bill_Gentoo"] = x["bill_length_mm"] * x["species_Gentoo"]
model.fit(x, y)

model.coef_
model.intercept_



# Y ~ X1 + X2
# Y = B0 + B1X1(부리길이) + B2X2(친스트랩) + B3X3(겐투)

# 곱해서 만든 인터랙션 항 => X1*X2, X1*X3
# Y = B0 + B1*X1 + B2*X2 + B3*X3 + B4*(X1*X2) + B5*(X1*X3)
# 즉, 단순히 B1만 있는 게 아니라
# 친스트랩 펭귄의 부리길이는 B1 + B4 만큼의 영향을 준다는 의미


# 곱셈 공식
# 아델리 회귀식
# B0 + B1X1 + 0 + 0 + 0 + 0

# 친스트랩 회귀식
# Y = B0 + B1X1 + B2 + 0 B4X1 + 0

# 겐투 회귀식
# Y = B0 + B3 + (B1 + B5)X1


# statmodels 사용한 분석과 시각화
# 팔머펭귄
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv"
penguins = pd.read_csv(url)
print(penguins.head())
np.random.seed(2022)
train_index = np.random.choice(penguins.shape[0], 200)
train_data = penguins.iloc[train_index]
train_data = train_data.dropna()
train_data.head()

# bill_depth_mm:species의 인터렉션 항을 포함한 선형회귀 모델 적합
model = ols("bill_length_mm ~ bill_depth_mm + species + bill_depth_mm:species",
             data=train_data).fit()
model.params
print(model.summary())
sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)
train_data['fitted'] = model.fittedvalues
# 산점도 (실제 데이터)
sns.scatterplot(data=train_data,
                x='bill_depth_mm', y='bill_length_mm',
                hue='species', palette='deep', edgecolor='w', s=50)
# 그룹별(facet별)로 fitted 선 그리기
for species, df in train_data.groupby('species'):
    df_sorted = df.sort_values('bill_depth_mm')  # X축 기준 정렬
    sns.lineplot(data=df_sorted,
                 x='bill_depth_mm', y='fitted')
plt.title("Regression Lines(fitted)")
plt.legend()
plt.show()
