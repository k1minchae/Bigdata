from shiny import render, ui
import plotly.express as px
from shiny.express import ui
from shiny.express import input
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shinywidgets import render_widget


# 팔머펭귄 데이터
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
penguins = load_penguins()

ui.input_selectize(
    "var",  
    "종을 선택하세요!",  
    choices=['Adelie', 'Gentoo', 'Chinstrap'],
)

ui.input_selectize(
    "x",
    "X축을 선택하세요!",
    choices=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
)

ui.input_selectize(
    "y",
    "Y축을 선택하세요!",
    choices=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
)

@render.plot
def plot():
    fig, ax = plt.subplots()
    ax.scatter(penguins.loc[penguins['species'] == input.var(), input.x()], penguins.loc[penguins['species'] == input.var(), input.y()])
    ax.set_xlabel(f'{input.x()}')
    ax.set_ylabel(f'{input.y()}')
    ax.set_title(f'{input.var()}: Bill Length vs Body Mass')
    return fig

@render.text
def select_value():
    return f"선택한 종은 {input.var()} 입니다."

with ui.sidebar():
    ui.input_select("species",
                    "종을 선택하세요.",
                    choices=['Adelie', 'Gentoo', 'Chinstrap'], selected='Adelie')

# shiny 기능 연습하기
# ui.panel_title("Hello Shiny!")
# ui.input_slider("n", "N", 0, 100, 20)

# ui.input_checkbox_group(  
#     "checkbox_group",  
#     "좋아하는 알파벳을 고르세요",  
#     {  
#         "a": "A가 좋아요",  
#         "b": "B가 좋아요",  
#         "c": "C가 좋아요",  
#     },  
# )  

# ui.input_date_range("daterange", "Date range", start=f"{datetime.now().date() - relativedelta(years=1)}")  

# @render.text
# def date_value():
#     return f"{input.daterange()[0]} 에서 {input.daterange()[1]} 까지"

# @render.text
# def value():
#     return ", ".join(input.checkbox_group())

# @render.text
# def txt():
#     return f"n*2 is {input.n() * 2}"
