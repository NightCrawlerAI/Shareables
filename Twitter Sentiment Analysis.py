#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
import pycountry
import re
import string
import time

from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


# Authentication
bearerToken = '<bearerToken>'
consumerKey = '<consumerKey>'
consumerSecret = '<consumerSecret>'
accessToken = '<accessToken>'
accessTokenSecret = '<accessTokenSecret>'

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth)
# api = tweepy.Client(bearerToken)#,consumerKey,consumerSecret,accessToken,accessTokenSecret)


# ### Sentiment Analysis

# In[3]:


api.rate_limit_status()


# In[4]:


#Sentiment Analysis
def percentage(part,whole):
    return 100 * float(part)/float(whole)

#Prompt User for input
keyword = input('Please enter keyword or hashtag to search: ')
noOfTweet = int(input('Please enter how many tweets to analyze: '))

# response = api.search_recent_tweets(query = keyword, max_results = 100)
# # print(response.meta)
# tweets = response.data
# # display(tweets)

#Pagination 
#Currently limited to search tweets from the past 7 days that will now need to be able to get the 
tweets = tweepy.Cursor(api.search_tweets, result_type = 'mixed', count = 100, q= keyword, tweet_mode = 'extended').items(noOfTweet)
positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []
post_dates = []



# for tweet in tweets:
# #     print(tweet.text)
# #     print(tweet.created_at)
#     tweet_list.append(tweet.text)
#     post_dates.append(tweet.created_at)
#     analysis = TextBlob(tweet.text)
#     score = SentimentIntensityAnalyzer().polarity_scores(tweet.text)
#     neg = score['neg']
#     neu = score['neu']
#     pos = score['pos']
#     comp = score['compound']
#     polarity += analysis.sentiment.polarity

#     if neg > pos:
#         negative_list.append(tweet.text)
#         negative += 1
#     elif pos > neg:
#         positive_list.append(tweet.text)
#         positive += 1
#     elif pos == neg:
#         neutral_list.append(tweet.text)
#         neutral += 1

for i, tweet in enumerate(tweets):
    tweet_list.append(tweet.full_text)
    post_dates.append(tweet.created_at)
    analysis = TextBlob(tweet.full_text)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet.full_text)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity

    if neg > pos:
        negative_list.append(tweet.full_text)
        negative += 1
    elif pos > neg:
        positive_list.append(tweet.full_text)
        positive += 1
    elif pos == neg:
        neutral_list.append(tweet.full_text)
        neutral += 1

    # Add a delay of 2.5 seconds between each request
    if i % 100 == 0:
        time.sleep(2.5)
        
        

positive = percentage(positive, noOfTweet)
negative = percentage(negative, noOfTweet)
neutral = percentage(neutral, noOfTweet)
polarity = percentage(polarity, noOfTweet)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')


# In[5]:


#Create a df of tweet id's to later be iterated over for magnitude analysis and their corresponding content
#Only works with 'Tweepy Client'
# tweet_ids = pd.DataFrame(tweets.items()).iloc[:,1:]

# api = tweepy.Client(bearerToken).search_recent_tweets(query = keyword, max_results = 100, next_token = True)
# api

# tweet_ids


#Texas media outlets scrape all 
#Times, Journal, FERC docs w/ Rachel, OSHA FMNCA energy only 
#LNG disruptions, freeport lng, pipeline explosion, power plants faults, 
#POC: Igor
#Teams Random
#Talk to Igor about building an algo that tracks market impact by historical event
#Event windows and statistical significance 

#The count of people to post on natural gas and when 
#Nymex natural gas
#Viral spread (taking into account the visibility by shares/followers)
#Correlate with Market moves
#Extract datetime data for interactions and their users 


# In[6]:


#Number of Tweets (Total, Positive, Negative, Neutral)
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
post_dates = pd.DataFrame(post_dates)

print('total number: ',len(tweet_list))
print('positive number: ',len(positive_list))
print('negative number: ', len(negative_list))
print('neutral number: ',len(neutral_list))


# In[7]:


tweet_list.insert(1, 'posted_date', post_dates)
# del tweet_list['Created At']
tweet_list.rename(columns = {
    0 : 'raw_tweet'
}, inplace = True)


# In[8]:


#Creating PieCart
labels = ['Positive ['+str((len(positive_list)/round((len(tweet_list)))*100))+'%]' , 'Neutral ['+str((len(neutral_list)/(len(tweet_list)))*100)+'%]','Negative ['+str((len(negative_list)/(len(tweet_list)))*100)+'%]']
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue','red']

patches, texts = plt.pie(sizes,colors=colors, startangle=90)

plt.style.use('default')
plt.legend(labels)
plt.title('Sentiment Analysis Result for keyword= '+keyword+'')
plt.axis('equal')

plt.show()


# In[9]:


tweet_list.drop_duplicates(subset = 'raw_tweet', inplace = True)


# In[10]:


#Cleaning Text (RT, Punctuation etc)

#Creating new dataframe and new features
tw_list = pd.DataFrame(tweet_list)
tw_list["text"] = tw_list['raw_tweet']

#Removing RT, Punctuation etc
remove_rt = lambda x: re.sub('RT @\w+: '," ",x)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
tw_list["text"] = tw_list.text.map(remove_rt).map(rt)
tw_list["text"] = tw_list.text.str.lower()
tw_list.head(10)


# In[43]:


tw_list.text[0]


# In[45]:


import tiktoken
import openai 

#Currently using Thomas' subscription
openai.api_key = "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"

def num_tokens(string, model):
    """Returns number of tokens that will be processed in a string.
    model: type of gpt model (str)
    (i.e. gpt-4, gpt-3.5-turbo, text-embedding-ada-002, text-davinci-002, text-davinci-003, davinci)"""
    
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(string))

num_tokens(tw_list.text[0], 'gpt-3.5-turbo')

messages = [
    {'role': 'system', 'content': 'You are an energy trading assistant.'},
    {'role': 'user', 'content': f'What does this mean: {tw_list.text[0]} '}
]

response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = messages
)

response


# In[58]:


print(response['choices'][0]['message']['content'])


# In[11]:


#Calculating Negative, Positive, Neutral and Compound values

tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in tw_list['text'].iteritems():
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    if neg > pos:
        tw_list.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        tw_list.loc[index, 'sentiment'] = "positive"
    else:
        tw_list.loc[index, 'sentiment'] = "neutral"
    tw_list.loc[index, 'neg'] = neg
    tw_list.loc[index, 'neu'] = neu
    tw_list.loc[index, 'pos'] = pos
    tw_list.loc[index, 'compound'] = comp

tw_list.head(10)


# In[12]:


tw_list.text[0]


# In[13]:


#Creating new data frames for all sentiments (positive, negative and neutral)
tw_list_negative = tw_list[tw_list['sentiment'] == 'negative']
tw_list_positive = tw_list[tw_list['sentiment'] == 'positive']
tw_list_neutral = tw_list[tw_list['sentiment'] == 'neutral']


# In[14]:


def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total', 'Percentage'])

#Count_values for sentiment
count_values_in_column(tw_list,'sentiment')


# In[15]:


# Create data for Donut Chart
pc = count_values_in_column(tw_list,'sentiment')
names= pc.index
size=pc['Percentage']
 
# Create a circle for the center of the plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(size, labels=names, colors=['green','blue','red'])
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[16]:


# tw_list.reset_index(drop = True).to_csv('boil_sentiment.csv', index = False)


# ### Wordcloud

# In[17]:


#Function to Create Wordcloud
def create_wordcloud(text, sentiment):
    mask = np.array(Image.open('cloud.png'))
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color='white',
    mask = mask,
    max_words=3000,
    stopwords=stopwords,
    repeat=True)
    wc.generate(str(text))
    wc.to_file('wc_' + sentiment + '.png')
    print('Word Cloud Saved Successfully')
    path='wc_' + sentiment + '.png'
    display(Image.open(path))


# In[18]:


#Creating wordcloud for all tweets
create_wordcloud(tw_list["text"].values, 'all')


# In[19]:


#Creating wordcloud for positive sentiment
create_wordcloud(tw_list_positive["text"].values, 'pos')


# In[20]:


#Creating wordcloud for negative sentiment
create_wordcloud(tw_list_negative["text"].values, 'neg')


# In[21]:


#Creating wordcloud for neutral sentiment
create_wordcloud(tw_list_neutral["text"].values, 'neutral')


# In[22]:


#Calculating tweet's lenght and word count
tw_list['text_len'] = tw_list['text'].astype(str).apply(len)
tw_list['text_word_count'] = tw_list['text'].apply(lambda x: len(str(x).split()))


# In[23]:


round(pd.DataFrame(tw_list.groupby("sentiment").text_len.mean()),2)


# In[24]:


round(pd.DataFrame(tw_list.groupby("sentiment").text_word_count.mean()),2)


# In[25]:


#Removing Punctuation
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

tw_list['punct'] = tw_list['text'].apply(lambda x: remove_punct(x))


# In[26]:


#Appliyng tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text

tw_list['tokenized'] = tw_list['punct'].apply(lambda x: tokenization(x.lower()))


# In[27]:


#Removing stopwords
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text
    
tw_list['nonstop'] = tw_list['tokenized'].apply(lambda x: remove_stopwords(x))


# In[28]:


#Appliyng Stemmer
ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))


# In[29]:


#Cleaning Text
def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove punctuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopword]  # remove stopwords and stemming
    return text


# In[30]:


tw_list.head()


# In[31]:


#Appliyng Countvectorizer
countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(tw_list['text'])
print('{} Number of reviews has {} words'.format(countVector.shape[0], countVector.shape[1]))
#print(countVectorizer.get_feature_names())


# In[32]:


countVector


# In[33]:


count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names_out())
count_vect_df.head()


# In[34]:


# Most Used Words
count = pd.DataFrame(count_vect_df.sum())
countdf = count.sort_values(0,ascending=False).head(20)
countdf[1:11].style.set_caption('Most Used Words')

