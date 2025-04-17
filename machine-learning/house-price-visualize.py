# 집값 시각화
import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv('./ames.csv')

df.head()

fig = go.Figure(go.Scattermapbox(
    lat=df["Latitude"],
    lon=df["Longitude"],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=10,               
        color=df["SalePrice"],
        colorscale='YlOrRd',
        opacity=0.6,
        colorbar=dict(       
            title="SalePrice", 
            x=1,             
            thickness=15,
            len=0.8
        )
    ),
    text=df["SalePrice"],
))

fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_zoom=11,
    mapbox_center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
    height=700,
    title="Ames House Price Visualization"
)

fig.show()