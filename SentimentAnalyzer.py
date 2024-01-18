import pandas as pd
import torch
from datetime import datetime
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np      
import json

class SentimentAnalyzer():
    
    def __init__(self, apikey, symbol):
        
        '''
        Description
        =======================================
        Module analyzing news and social media in terms of sentiment for given company
        
        Attributes
        ============================================
    
        apikey - apikey from https://site.financialmodelingprep.com/ account
        symbol -> str, company ticker e.g AAPL, MSFT
        '''
        self.apikey=apikey
        self.symbol=symbol
        with open ('valid_symbols_fmp.json') as f:
            self.symbols_list=json.load(f)
        try:
            with open('valid_symbols_fmp.json') as f:

                valid_symbols=json.load(f)

            if self.symbol not in valid_symbols:
                raise InvalidSymbol(self.symbol, valid_symbols)
        except FileNotFoundError:
            pass
        self.stock_news_sent_df=pd.DataFrame()
        self.social_sent_df=pd.DataFrame()
        
        self.pipe_stock_sent=pipeline('text-classification', model="slisowski/stock_sentiment_hp", tokenizer='ProsusAI/finbert', device='cpu', framework='pt')
        test_endpoint=f'https://financialmodelingprep.com/api/v3/profile/{self.symbol}?apikey={self.apikey}'
        response=requests.get(test_endpoint)
        if response.status_code==200:
            print('Connected')
        
        else:
            print(response.json())
        
        
    def __repr__(self):
        
        return f'Sentiment analyzer for {self.symbol}'
    
    
    
    def get_stock_news_sentiment(self, start_date):
        
        '''
        Description
        =======================================
        Method that acquire news data from FMP serverd
        
        Parameters
        ============================================
    
        start_date -> str date in format YYYY-mm-dd
        '''
        tokenizer = AutoTokenizer.from_pretrained("slisowski/stock_sentiment_hp")
        news_max_length=tokenizer.model_max_length
        start_date_dt=datetime.strptime(start_date, '%Y-%m-%d')
        news_sent_df=pd.DataFrame()
        index=0
        date_term_check=False
        
        months_years=[]
        
        endpoint=f'https://financialmodelingprep.com/api/v3/stock_news?limit=10000&tickers={self.symbol}&apikey={self.apikey}'
        data=requests.get(endpoint).json()
        self.plan_error(data)
        if len(data)==0:
            
            print(f'No news available for ticker {self.symbol}')
            return 0
        
        last_news_date=data[-1]['publishedDate']
        last_news_date_dt=datetime.strptime(last_news_date, '%Y-%m-%d %H:%M:%S')
        
        if last_news_date_dt>start_date_dt:
            
            print(f'Last available news is from {last_news_date}')
            
        
        
        for d in data:
            sentiment=[{'label': 'neutral', 'score': 0},
             {'label': 'negative', 'score': 0},
             {'label': 'positive', 'score': 0}]
            date_news=d['publishedDate']
            date_news_dt=datetime.strptime(date_news, '%Y-%m-%d %H:%M:%S')
            tokens=tokenizer.tokenize(d['text'])
            
                
                
            
            date_news_dt_control=datetime(year=date_news_dt.year, month=date_news_dt.month, day=1)

            if date_news_dt<start_date_dt:
                date_term_check=True
                break

            if  date_news_dt_control not in months_years:
                print(f'Calculating sentiment for data from {date_news_dt_control.month}.{date_news_dt_control.year}')
            months_years.append(date_news_dt_control)
            
            if len(tokens)>=news_max_length:
                sentiment=self.pipe_stock_sent(d['title'], top_k=3)
                
            else:
                sentiment=self.pipe_stock_sent(d['text'], top_k=3)
            for ds in sentiment:
                d[ds['label']]=ds['score']

            news_sent_df_temp=pd.DataFrame(d, index=[index])
            news_sent_df=pd.concat([news_sent_df, news_sent_df_temp])
            index=index+1
        
        
        news_sent_df['publishedDate']=pd.to_datetime(news_sent_df['publishedDate'])
        news_sent_df=news_sent_df.rename(columns={'publishedDate':'date'}).set_index('date')
        news_sent_df=news_sent_df.sort_index()
        self.stock_news_sent_df=news_sent_df
        

    def get_social_sentiment(self, start_date):
        
        '''
        Description
        =======================================
        Method that acquire twitter statistics from FMP serverd
        
        Parameters
        ============================================
    
        start_date -> str date in format YYYY-mm-dd
        '''
        
        
        endpoint=f'https://financialmodelingprep.com/api/v4/historical/social-sentiment?symbol={self.symbol}&limit=20000&apikey={self.apikey}'
        data=requests.get(endpoint).json()
        self.plan_error(data)
        if len(data)==0:
            print(f'No social sentiment statistics for ticker {self.symbol}')
            return 0
        start_date_dt=datetime.strptime(start_date, '%Y-%m-%d')
        
        social_sent_df=pd.DataFrame(data).dropna()
        last_sent_df_date_dt=datetime.strptime(social_sent_df['date'].iloc[-1], '%Y-%m-%d %H:%M:%S')
        if last_sent_df_date_dt>start_date_dt:
            print(f"Last available sentiment data are from {social_sent_df['date'].iloc[-1]}")
        social_sent_df['date']=pd.to_datetime(social_sent_df['date'])
        social_sent_df=social_sent_df.set_index('date')
        social_sent_df=social_sent_df.sort_index()
        social_sent_df=social_sent_df.loc[start_date:]
        self.social_sent_df=social_sent_df
            
            
    def plot_rsenti(self):
        
        '''
        Description
        =======================================
        Method that draw Relative Sentiment Index chart
        
        '''
        
        title=f'|RSentI |{self.symbol} |start_date={self.stock_news_sent_df.index[0]} |all news={len(self.stock_news_sent_df)}'
        if len(self.stock_news_sent_df)==0:
            print(f'Before plot sentiment score run get_stock_news_sentiment() with start_date arg')
            return 0
        elif 'RSentI' not in self.stock_news_sent_df.columns:
            print(f'Before plot sentiment score run calculate_rsenti() with window arg')
            return 0
        else:
            figure=go.Figure()
            figure.add_trace(go.Scatter(name='RSentI',x=self.stock_news_sent_df.index, y=self.stock_news_sent_df['RSentI'], mode='lines', marker_color='green'))
            figure.update_layout(title=title)
            figure.show()
       
    
    def calculate_rsenti(self, window=14):
        
        '''
        Description
        =======================================
        Method calculating Relative Sentiment Index
        
        Parameters
        ============================================
    
        window -> int, how many backward sentiment scores are taken into account when calculating
        
        '''
        if len(self.stock_news_sent_df)==0:
            print(f'Before calculate RSentI run get_stock_news_sentiment() with start_date arg')
            return 0
        
        if len(self.stock_news_sent_df)<=window:
            print(f'Not enough data {len(self.stock_news_sent_df)} to calculate RSentI for given window {window}, get data from longer period')
            return 0
        
        df=self.stock_news_sent_df.copy()
        df['sentiment_score']=df['positive']+(df['positive']*df['neutral'])-(df['negative'])-(df['negative']*df['neutral'])
        df['sent_pos']=np.where(df['sentiment_score']>=0,df['sentiment_score'],0)
        df['sent_neg']=np.where(df['sentiment_score']<0,abs(df['sentiment_score']),0)
        df=df.reset_index()
        df['calc_rsi_step_two']=np.where(df.index>13, True, False)
        df['sent_pos_mean']=df['sent_pos'].rolling(window).mean()
        df['sent_neg_mean']=df['sent_neg'].rolling(window).mean()
        df['sent_pos_mean_shift']=df['sent_pos_mean'].shift(1)
        df['sent_neg_mean_shift']=df['sent_neg_mean'].shift(1)
        math_formula_step_one=100-(100/(1+(df['sent_pos_mean']/df['sent_neg_mean'])))
        math_formula_step_two=100-(100/(1+(((df['sent_pos_mean_shift']*(window-1))+df['sent_pos'])/window)/(((df['sent_neg_mean_shift']*(window-1))+df['sent_neg'])/window)))
        df['RSentI']=np.where(df['calc_rsi_step_two']==False, math_formula_step_one, math_formula_step_two)
        cols_to_save=[col for col in df.columns if col not in ['sent_pos', 'sent_neg', 'sent_pos_mean','sent_neg_mean','sent_pos_mean_shift','sent_neg_mean_shift','calc_rsi_step_two']]
        df=df.loc[:,cols_to_save]
        df=df.set_index('date')
        self.stock_news_sent_df=df
        
    
    def calculate_socialma(self, window=20):
        
        '''
        Description
        =======================================
        Method calculating Social Moving Average
        
        Parameters
        ============================================
    
        window -> int, how many backward twitter sentiment scores are taken into account when calculating
        
        '''
        
        if len(self.social_sent_df)==0:
            print(f'Before calculate SocialMA run get_social_sentiment() with start_date arg')
            return 0
        
        if len(self.social_sent_df)<=window:
            print(f'Not enough data {len(self.social_sent_df)} to calculate RSentI for given window {window}, get data from longer period')
            return 0
        
        
        df=self.social_sent_df.copy()
        df['SocialMA']=df['stocktwitsSentiment'].rolling(window).mean()
        self.social_sent_df=df
        
    
    
    def plot_socialma(self):
        
        '''
        Description
        =======================================
        Method that draw Social Moving Average chart
        
        '''
        
        
        title=f'|SocialMA |{self.symbol} |start_date={self.social_sent_df.index[0]} |all scores={len(self.social_sent_df)}'
        if len(self.social_sent_df)==0:
            print(f'Before plot social score run get_social_sentiment() with start_date arg')
            return 0
        elif 'SocialMA' not in self.social_sent_df.columns:
            print(f'Before plot sentiment score run calculate_socialma() with window arg')
            return 0
        else:
            figure=go.Figure()
            figure.add_trace(go.Scatter(name='SocialMA',x=self.social_sent_df.index, y=self.social_sent_df['SocialMA'], mode='lines', marker_color='green'))
            figure.update_layout(title=title)
            figure.show()
    
    
    def plot_social_stats(self,stat_types, period=None):
        
        '''
        Description
        =======================================
        Method that draw twitter statistics on chart
        
        Parameters
        ============================================
    
        stat_types -> array with possible values {posts, comments, likes, impressions}
                period -> str, how resample data to plot, possible values are the same as for pandas.DataFrame().resample() method e.g. D,W,M or 1h, 2h ...
        
        '''

        
        
        if len(self.social_sent_df)==0:
            print(f'Before plot social stats run get_social_sentiment() with start_date arg')
            return 0
        
        stat_type_dict={'posts':'stocktwitsPosts', 'comments':'stocktwitsComments', 'likes':'stocktwitsLikes',
                       'impressions':'stocktwitsImpressions'}
        
        title=f'|{stat_types} from twitter |{self.symbol} |start_date={self.social_sent_df.index[0]} |all scores={len(self.social_sent_df)}'
        cols=[stat_type_dict[k] for k in  stat_type_dict.keys()]
        
        df_to_plot=self.social_sent_df.loc[:,cols]
        if period!=None:
            df_to_plot=df_to_plot.resample(period).sum()
        
        figure=None
        if len(stat_types)==2:
            figure=make_subplots(rows=2, cols=1, row_heights=[0.5,0.5], shared_xaxes=True,
                        vertical_spacing=0.01)
            figure.update_layout(height=1000)
            for i in range(len(stat_types)):
                figure.append_trace(go.Bar(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i]), row=i+1,col=1)
                figure.append_trace(go.Scatter(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i], mode='lines'), row=i+1,col=1)
            
            
        
        if (len(stat_types)>2)&(len(stat_types)<=4):
            figure=make_subplots(rows=2, cols=2, row_heights=[0.5,0.5], shared_xaxes=True,
                        vertical_spacing=0.01)
            figure.update_layout(height=1000)
            for i in range(len(stat_types)):
                if i>1:
                    figure.append_trace(go.Bar(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i]), row=i-1, col=2)
                    figure.append_trace(go.Scatter(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i], mode='lines'), row=i-1,col=2)
                else:
                    figure.append_trace(go.Bar(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i]), row=i+1, col=1)
                    figure.append_trace(go.Scatter(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i], mode='lines'), row=i+1,col=1)
                    
           
        if len(stat_types)==1:
            figure=make_subplots(rows=1, cols=1)
            figure.update_layout(height=600)
            
            figure.append_trace(go.Bar(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[0]]], name=stat_types[0]), row=1,col=1)
            figure.append_trace(go.Scatter(x=df_to_plot.index, y=df_to_plot[stat_type_dict[stat_types[i]]], name=stat_types[i], mode='lines'), row=1,col=1)
        figure.update_layout(title=title)
        figure.show()
    
    def plan_error(self,response):
        
        if type(response)==dict:
            if 'Error' in response.keys():
                print(response)
                return 0
            
        else:
            print('FMP API plan ok')
            pass
        
class InvalidSymbol(Exception):
    """Exception raised when invalid symbol occured.

    Attributes:
        symbol -- symbol which caused the error
        valid_symbols -- list of valid symbols
        message -- explanation of the error
    """

    def __init__(self, symbol, valid_symbols):
        self.valid_symbols=valid_symbols
        self.symbol= symbol
        self.message = f"Symbol {self.symbol} is invalid, choose from one of possible instruments:  {self.valid_symbols}"
        super().__init__(self.message)
