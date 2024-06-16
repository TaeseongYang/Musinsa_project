########## import ###########

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import random
import joblib 

from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

########## 홈페이지 이름 ###########

st.set_page_config(
    page_title='Musinsa_fit_service'
)

########## dashborad ###########

def dashboard():
    DATA_PATH = 'C:/Users/bigca/Toychang/kyonggi_project_2/'

    # review_ft = pd.read_excel(f'{DATA_PATH}review_ft.xlsx')
    review_ft = pd.read_excel(f'{DATA_PATH}review_6514.xlsx')
    review_ft = review_ft.loc[:, review_ft.columns != 'Review'].copy()

    main_features = ['총장', '어깨너비', '가슴단면', '소매길이']

    product = review_ft['Product_Name'].unique().tolist()
    selected_product = st.selectbox('**제품**', product)

    height = st.slider('키를 입력해주세요 (단위 : cm)', min_value=140, max_value=200, value=173)
    weight = st.slider('몸무게를 입력하세요 (단위 : kg)', min_value=30, max_value=100, value=63)
    
    low_error = random.randint(-2, 0)  # 키에 대한 오차 범위 
    high_error = random.randint(0, 2)  # 몸무게에 대한 오차 범위

    # 오차를 키와 몸무게에 적용
    min_height = height + low_error
    max_height = height + high_error
    min_weight = weight + low_error
    max_weight = weight + high_error

    st.markdown("""
        <div style='display: flex; align-items: center;'>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
            <i class="fas fa-robot" style='margin-right: 10px;'></i>
            <span>해당 옷 구매시 중점적으로 보는 핏을 체크해주세요!(**중복가능**)</span>
        </div>
    """, unsafe_allow_html=True)

    with st.container():
        check_length, check_shoulder, check_chest, check_sleeve = st.columns(len(main_features))
        with check_length:
            checked_length = st.checkbox('총장', value=True)
        with check_shoulder:
            checked_shoulder = st.checkbox('어깨너비', value=True)
        with check_chest:
            checked_chest = st.checkbox('가슴단면', value=True)
        with check_sleeve:
            checked_sleeve = st.checkbox('소매길이', value=True)
        get_selected_col = [checked_length, checked_shoulder, checked_chest, checked_sleeve]

    item = review_ft[(review_ft['Product_Name'] == selected_product) &
                     (min_height <= review_ft['Height']) & (review_ft['Height'] <= max_height) &
                     (min_weight <= review_ft['weight']) & (review_ft['weight'] <= max_weight)]
    
    item_reviews = len(item)
    no_reviews_features = item[main_features].isna().all()
    # 모든게 None일 경우에 True에 해당하는 것이다. 

    # main_features = ['총장','어깨너비','가슴단면','소매길이']
    main_features_filtered = [feature for feature, no_review in zip(main_features, no_reviews_features) if not no_review]

    selected_col = [main_features[i] for i, pick in enumerate(get_selected_col) if pick]
    review_counts = item[selected_col]

    sentiment_counts = review_counts.apply(lambda x: pd.Series(x.value_counts()), axis=0).fillna(0)

    # 결과보기 버튼 
    if st.button('결과보기'):
        fig = None  # fig 변수를 초기화합니다.

        try:
            if len(main_features_filtered) > 0:  # 조건에 맞는 리뷰가 있는 경우에만 실행
                fig = make_subplots(rows=1, 
                                    cols=len(main_features_filtered), 
                                    specs=[[{'type': 'domain'}] * len(main_features_filtered)],
                                    subplot_titles=main_features_filtered)
                
                colors = ['aqua', 'aquamarine', 'skyblue']

                fig.add_trace(go.Pie(labels = ['크다', '정핏', '작다'],
                                     values=[sentiment_counts['총장'].get('길다'),
                                             sentiment_counts['총장'].get('정핏'),
                                             sentiment_counts['총장'].get('짧다')],
                                     name = '총장',
                                     marker = dict(colors = colors)),1,1)

                fig.add_trace(go.Pie(labels = ['크다', '정핏', '작다'],
                                     values=[sentiment_counts['어깨너비'].get('크다'),
                                             sentiment_counts['어깨너비'].get('정핏'),
                                             sentiment_counts['어깨너비'].get('작다')],
                                     name = '어깨너비',
                                     marker = dict(colors = colors)),1,2)
                
                fig.add_trace(go.Pie(labels = ['크다', '정핏', '작다'],
                                     values=[sentiment_counts['가슴단면'].get('크다'),
                                             sentiment_counts['가슴단면'].get('정핏'),
                                             sentiment_counts['가슴단면'].get('작다'),],
                                     name = '가슴단면',
                                     marker = dict(colors = colors)),1,3)   

                fig.add_trace(go.Pie(labels = ['크다', '정핏', '작다'],
                                     values=[sentiment_counts['소매길이'].get('길다'),
                                             sentiment_counts['소매길이'].get('정핏'),
                                             sentiment_counts['소매길이'].get('짧다'),],
                                     name = '소매길이',
                                     marker = dict(colors = colors)),1,4)                              
                                    
                fig.update_traces(hole=.4, hoverinfo='label+percent+name')

                fig.update_layout(
                    title_text=selected_product + ' 리뷰 트렌드')
            else:
                st.error("당신의 키와 몸무게를 가진 리뷰는 없습니다!")
                st.error("키와 몸무게를 재설정해주세요!")
                st.write("")
                
        except ValueError as e:
            st.write("에러 발생:", e)
        
        if fig is not None:
            st.plotly_chart(fig)

########## size-prediction ###########

def size_prediction():
    # 저장경로 
    DATA_PATH = 'C:/Users/bigca/Toychang/kyonggi_project_2/'

    # 키와 몸무게 입력
    Height = st.number_input('키를 입력해 주세요! (단위 : cm)')
    Weight = st.number_input('몸무게를 입력해 주세요! (단위 : kg)')

    # 성별 선택 
    st.write('당신의 성별을 선택해 주세요!')
    Gender = st.radio("성별", ('남성', '여성'))

    if Gender == '남성':
        Gender = 0
    else:
        Gender = 1

    # 총장 선택
    st.write('당신이 원하는 총장 옵션을 선택해 주세요!')
    Length = st.radio("총장 옵션", ('길다', '정핏', '짧다'))

    if Length == '길다':
        Length = 1
    elif Length == '정핏':
        Length = 0
    elif Length == '짧다':
        Length = -1

    # 어깨너비 선택
    st.write('당신이 원하는 어깨너비 옵션을 선택해주세요!')
    Shoulder = st.radio("어깨너비 옵션", ('크다', '정핏', '작다'))

    if Shoulder == '크다':
        Shoulder = 1
    elif Shoulder == '정핏':
        Shoulder = 0
    elif Shoulder == '작다':
        Shoulder = -1

    # 가슴단면 선택
    st.write('당신이 원하는 가슴단면 옵션을 선택해주세요!')
    Chest = st.radio("가슴단면 옵션", ('크다', '정핏', '작다'))

    if Chest == '크다':
        Chest = 1
    elif Chest == '정핏':
        Chest = 0
    elif Chest == '작다':
        Chest = -1

    # 소매길이 선택
    st.write('당신이 원하는 소매길이 옵션을 선택해주세요!')
    Sleeve = st.radio("소매길이 옵션", ('길다', '정핏', '짧다'))

    if Sleeve == '길다':
        Sleeve = 1
    elif Sleeve == '정핏':
        Sleeve = 0
    elif Sleeve == '짧다':
        Sleeve = -1

    # 선택박스 가로로 뒤집기
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # 예측 버튼 추가
    if st.button('예측하기'):
        # 새로운 데이터 프레임 만들기
        df = pd.DataFrame({
            'Height': [Height],
            'Weight': [Weight],
            '총장': [Length],
            '어깨너비': [Shoulder],
            '가슴단면': [Chest],
            '소매길이': [Sleeve],
            'Gender': [Gender]
        })

        # 저장된 모델 가중치 불러오기 
        rf_Length = joblib.load(f'{DATA_PATH}rf_Length.pkl')
        rf_Shoulder = joblib.load(f'{DATA_PATH}rf_Shoulder.pkl')
        rf_Chest = joblib.load(f'{DATA_PATH}rf_Chest.pkl')
        rf_Sleeve = joblib.load(f'{DATA_PATH}rf_Sleeve.pkl')

        # 저장된 스케일러 가중치 불러오기 
        rob_scaler = joblib.load(f'{DATA_PATH}robust_scaler.pkl')

        # 스케일 작업 실시 
        df_scaled = rob_scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

        # 예측작업 실시 
        pre_Length = rf_Length.predict(df_scaled)
        pre_Shoulder = rf_Shoulder.predict(df_scaled)
        pre_Chest = rf_Chest.predict(df_scaled)
        pre_Sleeve = rf_Sleeve.predict(df_scaled)

        text = f"고객님의 신체 정보 {Height}cm / {Weight}kg를 기준으로 저희 예측 모델에 의하면 다음과 같은 추천 사이즈가 예측됩니다."
        text_info = f"""
         - 총장: {round(pre_Length[0],2)} 
         - 어깨너비: {round(pre_Shoulder[0],2)} 
         - 가슴단면: {round(pre_Chest[0],2)} 
         - 소매길이: {round(pre_Sleeve[0],2)} 
         """

        st.markdown(f"""
            <div style='display: flex; align-items: center;'>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">
                <i class="fas fa-robot" style='margin-right: 10px;'></i>
                <span>{text}</span>
            </div>
        """, unsafe_allow_html=True)
        st.markdown(text_info)



               
########### 좌측 사이드바 #############
class Multiapp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, function):
        self.apps.append({
            'title': title,
            'function': function
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Musinsa',
                options=['DashBorad', 'Size-prediction'],
                icons=['reception-3', 'reception-4'],
                menu_icon='chat-text-fill',
                default_index = 0,
                styles={
                    'container': {'padding': '5!important', 'background-color': 'black'},
                    'icon': {'color': 'white', 'font-size': '23px'},
                    'nav-link': {'color': 'white', 'font-size': '20px', 'text-align': 'left', 'margin': '0px'},
                    'nav-link-selected': {'background-color': '#02ab21'}
                }
            )

        if app == 'DashBorad':
            dashboard()

        elif app == 'Size-prediction':
            size_prediction()

multiapp = Multiapp()
multiapp.add_app("DashBorad", dashboard)
multiapp.add_app("Size-prediction", size_prediction)
multiapp.run()