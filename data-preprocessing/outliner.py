# 데이터 전처리
# 5. 이상치 처리 방법
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warpbreaks = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/warpbreaks.csv')

# 1. BoxPlot 활용
warpbreaks.boxplot(column = ['breaks']);
plt.show();

# 1분위수 계산
Q1 = np.quantile(warpbreaks['breaks'], 0.25)
# 3분위수 계산
Q3 = np.quantile(warpbreaks['breaks'], 0.75)
IQR = Q3 - Q1

UC = Q3 + (1.5 * IQR) # 위 울타리
LC = Q3 - (1.5 * IQR) # 위 울타리

print(warpbreaks.loc[(warpbreaks.breaks > UC) | (warpbreaks.breaks < LC), :])

# 2. Z-Score 활용
upper = warpbreaks['breaks'].mean() + (3*warpbreaks['breaks'].std())
lower = warpbreaks['breaks'].mean() - (3*warpbreaks['breaks'].std())

warpbreaks.loc[(warpbreaks.breaks > upper) | (warpbreaks.breaks < lower), :].head(3)

# 이상치 여부 컬럼 추가
warpbreaks['z_outlier'] = warpbreaks['breaks'].apply(lambda x: 'Outlier' if x > upper or x < lower else 'Normal')

# 이상치 시각화
plt.figure(figsize=(10, 6))
sns.stripplot(data=warpbreaks, x='wool', y='breaks', hue='z_outlier', 
              palette={'Outlier': 'red', 'Normal': 'gray'}, jitter=True)
plt.axhline(upper, color='blue', linestyle='--', label='Z-score Upper Bound')
plt.axhline(lower, color='blue', linestyle='--', label='Z-score Lower Bound')
plt.title('Z-Score Outlier Detection')
plt.legend()
plt.show()