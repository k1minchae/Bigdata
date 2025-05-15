# 군집 분석
# 비지도학습(Unsupervised Learning)의 대표 기법
# 레이블 없이 유사한 특성을 가진 데이터끼리 그룹화

# 마케팅, 고객 세분화, 이상탐지 등 다양한 분야에 활용

'''
비지도학습인 군집분석도 군집분석이 잘 수행되었는지 평가해야함

1. 내부 유효성 지표
2. 외부 유효성 지표
    : 임의의 관측치가 어떤 군집에 속하는지 알고 있는 경우
    - Rand Index : accuracy와 유사한 개념
    - Adjusted Rand Index : 군집의 개수에 따라 조정된 Rand Index

'''

# 외부 유효성 지표 예시
# 1. rand index
from sklearn.metrics import confusion_matrix
y_true = [1, 0, 1, 1, 1, 1, 2]  # 실제값
y_pred = [1, 1, 0, 0, 1, 1, 2]  # 예측값

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# rand index 계산
from sklearn.metrics.cluster import rand_score
rand_score(y_true, y_pred)

# sklearn 을 활용하여 accuracy 계산
# 계산 결과를 비교해보면 rand index와 accuracy는 동일한 결과를 보임
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)


# 2. adjusted rand index
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y_true, y_pred)     # -0.5 ~ 1 사이의 값
# 군집의 수가 늘어날수록 값이 커짐



##################################################################

# 내부 유효성 지표
# 임의의 관측치가 어떤 군집에 속하는지 알 수 없는 경우
# 1. 응집도
# 군집 내의 관측치 간의 거리

# 2. 분리도
# 군집 간의 거리


# 실루엣 계수
# 군집 내의 응집도와 군집 간의 분리도를 동시에 고려
# 클수록 성능이 좋다.

'''
실루엣 계수 예시
'''
# 1. 데이터 불러오기
import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')
print(df.head(2))

# 2. 데이터 표준화
# 군집 분석은 표준화를 전체 데이터에 해도됨
# target이 없기 때문에
from sklearn.preprocessing import StandardScaler
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))

# 3. 이상치 제거
# EDA 를 통해 제거한다.

# 4. K-means clustering
# 각 관측치별로 번호를 붙여서 
# - 사전에 값을 지정해줘야 함
# - 이상치에 민감
# - 밀도가 다양한 데이터의 경우 성능이 떨어짐

from sklearn.cluster import KMeans # K-평균 군집분석 불러오기
kmeans = KMeans(n_clusters = 4, random_state = 1)
labels = kmeans.fit_predict(df_trans)
print(labels)  # 각 관측치별로 군집 번호가 붙음


'''
군집 분석 전에 표준화를 진행해줘야함
군집 알고리즘 특성상 스케일에 민감함.

이상치도 제거하는것이 좋다.
군집분석은 이상치에 민감함
'''
#####################################################################

# 군집의 수 정하기
# 팔꿈치 방법(Elbow method)
# 군집 내 총 변동은 .intertia로 확인 가능

# 군집의 수를 늘려가면서 inertia를 확인
kmeans.inertia_ # 군집 내 총 변동

# k = 2~10 까지 반복
from sklearn.cluster import KMeans
wss = [] # 계산 결과를 저장할 빈 리스트 생성

for i in range(2, 10):
    fit_kmeans = KMeans(n_clusters=i, random_state=11).fit(df_trans) # k별 군집분석 수행
    wss.append(fit_kmeans.inertia_) # k별 .inertia_ 결과 wss에 저장

print(wss)


# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(11,8.5))
plt.plot(range(2,10), wss, 'bx-');
plt.xlabel('Number of clusters $k$');
plt.ylabel('Inertia');
plt.title('The Elbow Method showing the optimal $k$');
plt.show();


#################################################################
# 실루엣 계수로 군집의 수 정하기
# 평균 실루엣 계수가 가장 큰 군집의 수를 선택
from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
scores = [] # 계산 결과를 저장할 빈 리스트 생성
for i in range(2,10):
    fit_kmeans = KMeans(n_clusters=i, random_state=11).fit(df_trans) # k별 군집분석 수행
    score = silhouette_score(df, fit_kmeans.labels_) # k별 실루엣 계수 계산
    scores.append(score) #k별 실루엣 계수 저장
    print("For n_clusters={0}, the silhouette score is {1}".format(i, score))

plt.plot(range(2,10), np.array(scores), 'bx-');
plt.xlabel('Number of clusters $k$');
plt.ylabel('Average Silhouette');
plt.title('Average Silhouette method showing the optimal $k$');
plt.show()

# k = 2 일 때 실루엣 계수가 가장 큰 것을 확인할 수 있음


# 실루엣계수를 통해 k-means 군집 분석 수행
kmeans = KMeans(n_clusters = 2)
labels = kmeans.fit_predict(df_trans)
df['cluster_label'] = labels
print(df.head(2))

# 실루엣 계수: 관측치 하나당 거리를 계산 => 데이터가 많으면 비효율적


##################################################################

# 계층적 군집 분석
# 1. 완전연결: 군집간 관측치의 쌍별 최대거리 이용 -> 병합
# 2. 단일연결: 군집간 관측치의 쌍별 최소거리 이용
# 3. 평균연결: 군집간 관측치의 쌍별 평균거리 이용
# 4. 중심연결: 군집의 중심간 거리 이용
# 5. Ward 연결: 군집 내 분산의 증가를 최소화하는 방식으로 군집 병합

# 예제
# 1. distance matrix 계산
# 2. 가장 가까운 관측치 쌍 선택
# 3. distance matrix 업데이트
# 4. 가장 가까운 관측치 쌍 선택
# 5. 군집의 수가 1이 될 때까지 반복

from sklearn.cluster import AgglomerativeClustering

import pandas as pd
import numpy as np
df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/USArrests.csv')

# 전처리
from sklearn.preprocessing import StandardScaler # 표준화 전처리 모듈 불러오기
numeric_data = df.select_dtypes('number')
stdscaler = StandardScaler()
df_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)
print(df_trans.head(2))

# 적절한 군집의 수 계산 (실루엣)
from sklearn.metrics import silhouette_score
scores = []
for i in range(2,10):
    fit_hk = AgglomerativeClustering(n_clusters=i, linkage = 'ward').fit(df_trans)
    score = silhouette_score(df_trans, fit_hk.labels_)
    scores.append(score)

import matplotlib.pyplot as plt
plt.figure(figsize=(11,8.5));
plt.plot(range(2,10), np.array(scores), 'bx-');
plt.xlabel('Number of clusters $k$');
plt.ylabel('Average Silhouette');
plt.title('Average Silhouette method showing the optimal $k$');
plt.show();

# 계층적 군집 분석 수행
hk = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
hk.fit(df)

label = hk.labels_

df2 = df.copy()
df2['cluster'] = label
df2.head(5)

# 덴드로그램 시각화
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(df, method="ward")
fig = plt.figure(figsize=(10, 6));
dendrogram(Z);
plt.show();


###################################################################

# DBSCAN
# 밀도 기반 군집 분석
# 밀도가 높은 지역을 군집으로 정의
# 반경 설정

# 데이터 불러오기
import pandas as pd
import numpy as np
df_8 = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/s14_df.csv')

# 데이터 전처리
from sklearn.preprocessing import StandardScaler
numeric_data = df_8.select_dtypes('number')
stdscaler = StandardScaler()
df_8_trans = pd.DataFrame(stdscaler.fit_transform(numeric_data), columns = numeric_data.columns)

from sklearn.cluster import DBSCAN
# MinPts: 군집으로 묶기 위한 최소 관측치 수
# eps: 반경

# MinPts 정하는 법 : 2 x (차원 수)
# eps 정하는 법 : k-최근접 이웃 그래프를 통해 결정
# 엘보우 포인트를 찾아서 eps 결정

from sklearn.neighbors import NearestNeighbors
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(df_8_trans)
distances, indices = neighbors_fit.kneighbors(df_8_trans)


# eps 결정
import matplotlib.pyplot as plt
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.axhline(y=0.15, color='black', linestyle='--', linewidth=3)
plt.show();
# eps = 0.15


# DBSCAN 학습
db = DBSCAN(eps=0.15, min_samples=4).fit(df_8)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
# 군집의 수는 5, noise point는 총 29개 관측된 것을 확인할 수 있습니다.

df_8['cluster_label'] = labels
plt.figure(figsize=(8, 6))
plt.scatter(x = df_8['x'], # x축
            y = df_8['y'], # y축
            s=30, # 점 크기
            c=df_8['cluster_label']) # 점 color
plt.xlabel('x');
plt.ylabel('y');
plt.show()