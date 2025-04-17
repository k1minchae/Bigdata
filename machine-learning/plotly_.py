import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

df = pd.read_csv('../practice-data/house-data/train.csv')

y = df['SalePrice']
x = np.linspace(min(y), max(y))

px.scatter(
    x = 'MSSubClass',
    y = 'SalePrice',
    data_frame = df,
    title='집값데이터 산점도'
)


# 비트코인 차트 그리기
import ccxt

binance = ccxt.binance()
btc_ohlcv = binance.fetch_ohlcv('BTC/USDT', '1d', limit=1000)

df = pd.DataFrame(btc_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
df.set_index('datetime', inplace=True)

import plotly.graph_objects as go
fig = go.Figure();
fig.add_trace(
    go.Scatter(
        x=df.index, # X축: 날짜
        y=df['close'], # Y축: 종가
        mode='lines+markers', # 선 + 마커 같이 표시
        marker=dict(
            color='blue', # 마커 색상
            size=6, # 마커 크기
            symbol='circle' # 원형 마커
    ),
    line=dict(
        color='blue', # 선 색상
        width=2, # 선 두께
        dash='solid' # 실선 스타일
    ),
        name="BTC/USDT Closing Price"
    )
);
fig.show()


# 강사님코드

lcd_df = pd.read_csv('./data/plotlydata/seoul_bike.csv')
lcd_df.shape
import plotly.express as px
# 지도 기반 산점도 생성
fig = px.scatter_mapbox(
    lcd_df,
    lat="lat",                 # 위도
    lon="long",                # 경도
    size="LCD거치대수",         # 원 크기
    color="자치구",             # 색상 구분 기준
    hover_name="대여소명",       # 마우스 오버 시 주요 텍스트
    hover_data={
        "lat": False,
        "long": False,
        "LCD거치대수": True,
        "자치구": True
    },
    text="text",               # 지도에 직접 표시될 텍스트
    zoom=11,                   # 줌 레벨
    height=650                 # 그래프 높이
)
# 지도 스타일 및 여백 설정
fig.update_layout(
    mapbox_style="carto-positron",  # 배경 지도 스타일 (무료)
    margin={"r": 0, "t": 0, "l": 0, "b": 0}  # 여백 제거
)
# 지도 시각화 출력
fig.show()
pd.set_option('display.max_columns', None)
import geopandas as gpd
gdf = gpd.read_file("./data/plotlydata/서울시군구/TL_SCCO_SIG_W.shp")
gdf.head(7)
gdf.shape
gdf.info()
gdf["geometry"][0]
gdf["geometry"][1]
print(gdf.crs)
gdf = gdf.to_crs(epsg=4326)
gdf.to_file("./data/plotlydata/seoul_districts.geojson",
             driver="GeoJSON")