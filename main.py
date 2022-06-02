import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()



@st.cache
def get_data(filename):
    annual_data = pd.read_csv(filename)
    return annual_data

with header:
    st.title("My first streamlit project!!")
    st.text('In this project I look into the transactions of taxis in NYC. ...')

with dataset:
    st.header('NYC Taxi dataset')
    st.text('I found this dataset on google.com, ...')

    annual_data = get_data('data/annual-enterprise-survey-2020-financial-year-provisional-csv.csv')

    st.subheader('Pick-up loation ID distribution on the NYC dataset')
    value_dist = pd.DataFrame(annual_data['Value'].value_counts().head(50))
    st.bar_chart(value_dist)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
    st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic..')


with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performace changes')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should be there?', options=[100,200,300,'No limit'], index=0)

    input_feature = sel_col.text_input('Which features should be used as the inpt feature?', 'Value')

    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    disp_col.subheader("disp col")
