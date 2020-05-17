##############################################################
# Customer Issue Prediction Model
#-------------------------------------------------------------
# Author : Alisa Ai
#
##############################################################
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go
from plotly import tools
from chart_studio import plotly

import csv
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

import dill

##############################################################

app = dash.Dash()

#app.layout = html.Div(children=[
    #html.H1(children='Predict Customer Issues', style={'textAlign': 'center'}),

    #html.Div(children=[
        #html.Label('Enter you complaints: '),
        #dcc.Input(id='complaints-text', placeholder='Complaints', type='text'),
        #html.Div(id='result')
    #], style={'textAlign': 'center'}),
app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

layout = dict(
    autosize=True,
    height=450,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=45,
        r=15,
        b=45,
        t=35
    )
)

app.layout = html.Div([
    # Title - Row
    html.Div(
        [
            html.H1(
                'Customer Issue Prediction App',
                style={'font-family': 'Helvetica',
                       "margin-left": "20",
                       "margin-bottom": "0"},
                className='eight columns',
            )
        ],
        className='row'
    ),

    #block 2
    html.Div([
        dcc.Store(id = 'memory'),
        html.Div(
            [
                html.Div(
                    [
                        html.Label('Enter your complaints here: '),
                        dcc.Input(id='complaints-text', placeholder='Complaints', type='text',
                        style=dict(width='1000px', height='100px', display='inline-block', verticalAlign="middle"))],
                        className='eight columns',
                        style={"height": "auto", "width": "2000px", "margin-bottom": "auto", 'whiteSpace': 'pre-line'   }
                        ),
                html.Div(
                    [
                        html.P('Select Your Product:'),
                        dcc.Dropdown(id = 'product', options=[
                                    {'label': 'Checking or savings account', 'value': 1},
                                    {'label': 'Consumer Loan', 'value': 2},
                                    {'label': 'Credit card or prepaid card', 'value': 3},
                                    {'label': 'Credit reporting, credit repair services, or other personal consumer reports', 'value': 4},
                                    {'label': 'Debt collection', 'value': 5},
                                    {'label': 'Money transfer, virtual currency, or money service', 'value': 6},
                                    {'label': 'Mortgage', 'value': 7},
                                    {'label': 'Other financial service', 'value': 8},
                                    {'label': 'Payday loan, title loan, or personal loan', 'value': 9},
                                    {'label': 'Student loan', 'value': 10},
                                    {'label': 'Vehicle loan or lease', 'value': 11}],
                                placeholder="Select Your Product",
                                style=dict(width='300px', height='40px', display='inline-block', verticalAlign="middle"))],
                        className='three columns',
                        style={"height": "auto", "width": "2000px", "margin-bottom": "auto"}
                        ),
                html.Div(
                    [
                        html.P('Select Your Company:'),
                        dcc.Dropdown(
                                id = 'company', options=[
                                    {'label': 'CAPITAL ONE', 'value': 12},
                                    {'label': 'CITIBANK', 'value': 13},
                                    {'label': 'EQUIFAX', 'value': 14},
                                    {'label': 'Experian Information Solutions', 'value': 15},
                                    {'label': 'JPMORGAN CHASE & CO.', 'value': 16},
                                    {'label': 'Navient Solutions', 'value': 17},
                                    {'label': 'SYNCHRONY FINANCIAL', 'value': 18},
                                    {'label': 'TRANSUNION INTERMEDIATE HOLDINGS', 'value': 19},
                                    {'label': 'WELLS FARGO & COMPANY', 'value': 20},
                                    {'label': 'Other', 'value': 21}],
                                placeholder="Select Your Company",
                                style=dict(width='300px', height='40px', display='inline-block', verticalAlign="middle"))],
                        className='three columns',
                        style={"height": "auto", "width": "2000px", "margin-bottom": "auto"}
                        ),
                html.Div(
                    [
                        html.P('Select Your State:'),
                        dcc.Dropdown(
                                id = 'state', options=[
                                    {'label': 'FL', 'value': 22},
                                    {'label': 'GA', 'value': 23},
                                    {'label': 'IL', 'value': 24},
                                    {'label': 'NC', 'value': 25},
                                    {'label': 'NJ', 'value': 26},
                                    {'label': 'NY', 'value': 27},
                                    {'label': 'OH', 'value': 28},
                                    {'label': 'PA', 'value': 29},
                                    {'label': 'TX', 'value': 30},
                                    {'label': 'Other', 'value': 31}],
                                placeholder="Select Your State",
                                style=dict(width='300px', height='40px', display='inline-block', verticalAlign="middle"))],
                        className='three columns',
                        style={"height": "auto", "width": "2000px", "margin-bottom": "auto"}
                        ),
                html.Div(
                    [
                        html.Button('Submit', id='button_1')
                    ],
                    className='one columns',
                    style={'margin-bottom': 'auto'}
                ),
                html.Div(id='result')],
                style={'textAlign': 'center'})
            ])
        ])

@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='complaints-text', component_property='value'),
    Input(component_id='product', component_property='value'),
    Input(component_id='company', component_property='value'),
    Input(component_id='state', component_property='value'),
    Input('button_1', 'n_clicks')]
)
def update_issue(complaints, pro, comp, stat, n_clicks):
    if n_clicks is not None:
        if complaints is not None and complaints is not '':
            try:
                ############# vaderSentiment
                text = re.sub("[XX$]"," ", complaints)
                text = re.sub(r'\s+', ' ', text)
                analyser = SentimentIntensityAnalyzer()
                pos = analyser.polarity_scores(text)['pos']
                neg = analyser.polarity_scores(text)['neg']
                ############# Clean
                text2 = re.sub("[^a-zA-Z]"," ", text)
                stopword = set(stopwords.words('english'))
                text2 = ' '.join([word for word in text2.split() if word not in (stopword)])
                porter_stemmer = PorterStemmer()
                text2 = porter_stemmer.stem(text2)
                ############# input organize
                index_dict = {
                'Product_Checking or savings account': 1,
                'Product_Consumer Loan': 2,
                'Product_Credit card or prepaid card': 3,
                'Product_Credit reporting, credit repair services, or other personal consumer reports': 4,
                'Product_Debt collection': 5,
                'Product_Money transfer, virtual currency, or money service': 6,
                'Product_Mortgage': 7,
                'Product_Other financial service': 8,
                'Product_Payday loan, title loan, or personal loan': 9,
                'Product_Student loan': 10,
                'Product_Vehicle loan or lease': 11,
                'Company_CAPITAL ONE FINANCIAL CORPORATION': 12,
                'Company_CITIBANK, N.A.': 13,
                'Company_EQUIFAX, INC.': 14,
                'Company_Experian Information Solutions Inc.': 15,
                'Company_JPMORGAN CHASE & CO.': 16,
                'Company_Navient Solutions, LLC.': 17,
                'Company_SYNCHRONY FINANCIAL': 18,
                'Company_TRANSUNION INTERMEDIATE HOLDINGS, INC.': 19,
                'Company_WELLS FARGO & COMPANY': 20,
                'Company_Other': 21,
                'State_FL': 22,
                'State_GA': 23,
                'State_IL': 24,
                'State_NC': 25,
                'State_NJ': 26,
                'State_NY': 27,
                'State_OH': 28,
                'State_PA': 29,
                'State_TX': 30,
                'State_Other': 31}
                def dummy(index_dict, pro, comp, stat):
                    for key, value in index_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                        if pro == value:
                            index_dict[key] = 100
                        if comp == value:
                            index_dict[key] = 100
                        if stat == value:
                            index_dict[key] = 100
                    for key, value in index_dict.items():
                        if value < 100:
                            index_dict[key] = 0
                        if value == 100:
                            index_dict[key] = 1
                    return index_dict

                attribute_index = dummy(index_dict=index_dict, pro=pro, comp=comp, stat=stat)
                attribute_index['positive_score'] = pos
                attribute_index['negative_score'] = neg
                attribute_index['clean_sentences'] = 'text2'
                input_data = pd.DataFrame(attribute_index, index=[0])
                issue = model.predict(input_data)[0]
                return 'Are you facing with this issue: {}?'.format(str(issue))
            except ValueError:
                return 'Unable to predict issue'

if __name__ == '__main__':
    with open('/Users/hengyuai/Documents/QMSS_1/PD/Customer-Issue-Prediction/src/pipeline.pkl', 'rb') as file:
        model = dill.load(file)
    app.run_server(debug=True)
