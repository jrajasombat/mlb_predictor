import rebalancing_functions_st as reb
import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
import pybaseball as pbb
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Set page name and icon
st.set_page_config(
    page_title = 'MLB Predictor',
    page_icon = 'n.png',
)

# Set page title
st.title('Predicting MLB winning probabilities')
st.markdown('## ')

# Initiate sidebar
st.sidebar.markdown('## Select Parameters')

# Enter inputs
your_team = st.sidebar.selectbox('Select your team', (
'ARI',
'ATL',
'BOS',
'CHC',
'CHW',
'CIN',
'CLE',
'COL',
'DET',
'HOU',
'KCR',
'LAA',
'LAD',
'MIA',
'MIL',
'MIN',
'NYM',
'NYY',
'OAK',
'PHI',
'PIT',
'SDP',
'SEA',
'SFG',
'STL',
'TBR',
'TEX',
'TOR',
'WSN'
))

your_home = st.sidebar.selectbox('Are they playing at home?', ('Yes','No'))

your_pitcher = st.sidebar.text_input('Who is their starting pitcher?')

opp_team = st.sidebar.selectbox('Select the opposing team', (
'ARI',
'ATL',
'BOS',
'CHC',
'CHW',
'CIN',
'CLE',
'COL',
'DET',
'HOU',
'KCR',
'LAA',
'LAD',
'MIA',
'MIL',
'MIN',
'NYM',
'NYY',
'OAK',
'PHI',
'PIT',
'SDP',
'SEA',
'SFG',
'STL',
'TBR',
'TEX',
'TOR',
'WSN'
))

opp_pitcher = st.sidebar.text_input('Who is their starting pitcher?')

your_odds = st.sidebar.number_input('Enter decimal odds for your team', min_value = 1.00)
opp_odds = st.sidebar.number_input('Enter decimal odds for the opposing team', min_value = 1.00)



#reb.rebalance(coi)

# Configure sidebar
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## Discussion')
st.sidebar.markdown('* One approach to reduce risk when constructing an investment portfolio \
is __maximum diversification portfolio optimization__. In particular, this strategy maximizes \
a diversification ratio and does not take into account expected returns.')
st.sidebar.markdown('* The __diversification ratio__ is defined as the ratio of the weighted average \
of all the volatilites in the portfolio divided by the total portfolio volatility.')
st.sidebar.markdown('* This web app optimizes a portfolio of stocks, bonds, and gold at regular \
intervals.')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')


# Logo in the sidebar
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}" target="_blank">
            <img src="data:image/{img_format};base64,{bin_str}" width = "75" />
        </a>'''
    return html_code

png_html = get_img_with_href('n.png', 'https://www.jimisinith.com')

col1, col2, col3 = st.sidebar.beta_columns([3,7,1])
with col1:
    st.write("")
with col2:
    st.markdown(png_html, unsafe_allow_html=True)
with col3:
    st.write("")
