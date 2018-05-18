import tweepy
from tweepy import OAuthHandler
import json
import pymongo
from pymongo import MongoClient
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import numpy as np
Consumer_key = 'mTvWV73JZmHZISK4K3lQ6ee68'
Secret_consumer = 'Eqwq9BEP0HaGcAoHYHLqk6zWkEvgQ67ZRpIg8Vz80eUN0bouT4'
Token_secret = 'wI149iuOeq0SOaUUngjpayuMWqQAQMHSZqiSp7rfXSyjg'
Access_token = '2429282838-cIjn961OgjZF4O4VCvKQ7Nmp2noRjGhykrPJ03x'
auth = OAuthHandler(Consumer_key, Secret_consumer)
auth.set_access_token(Access_token, Token_secret)

def twitter_connect():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(Consumer_key, Secret_consumer)
    auth.set_access_token(Access_token, Token_secret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

extractor = twitter_connect()

# We create a tweet list as follows
tweets = extractor.user_timeline(screen_name="EASPORTSFIFA", count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:10]:
    print(tweet.text)
    print()	
	

twitter_data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])


import textblob
from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'
		

twitter_data['length']  = np.array([len(tweet.text) for tweet in tweets])
twitter_data['Tweet_ID']   = np.array([tweet.id for tweet in tweets])
twitter_data['No_Likes']  = np.array([tweet.favorite_count for tweet in tweets])
twitter_data['Retweets']    = np.array([tweet.retweet_count for tweet in tweets])		
twitter_data['Date_Posted'] = np.array([tweet.created_at for tweet in tweets])
twitter_data['Source'] = np.array([tweet.source for tweet in tweets])

twitter_data['sentiments'] = np.array([ analyze_sentiment(tweet) for tweet in twitter_data['Tweets']])
print(twitter_data)

