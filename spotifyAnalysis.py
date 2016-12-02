# Databricks notebook source exported at Fri, 2 Dec 2016 03:19:32 UTC
# Commands run by nvk
# MAGIC %md
# MAGIC # Spotify Data Analysis

# COMMAND ----------

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import time
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC * Loading the "user_data_sample.csv" in to a pandas dataframe

# COMMAND ----------

users_df = pd.read_csv("https://s3-us-west-1.amazonaws.com/vamsinallabothubucket/user_data_sample.csv")
users_df.describe()

# COMMAND ----------

users_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Loading the end_song_sample.csv from AWS into another pandas dataframe.

# COMMAND ----------

songs_df = pd.read_csv("https://s3-us-west-1.amazonaws.com/vamsinallabothubucket/end_song_sample.csv")

# COMMAND ----------

songs_df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC * Merging both the song and user dataframes into a single dataframe where the user_id matches in both the data drames

# COMMAND ----------

combined_df = pd.merge(users_df,songs_df,on='user_id')
combined_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * The combined dataframe contains a total of 1342891 rows and 10 columns

# COMMAND ----------

combined_df.shape # number of rows and columns
#combined_df.drop(['column1', 'column2', 'column3'], axis=1, inplace=True) <- for removing the rows and columns from the  dataframe
#axis = 1 for columns and axis = 0 for rows

# COMMAND ----------

combined_df.age_range.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC * finding the total number of tracks played and minutes(60000 millisec/min) listened and number of users under each gender

# COMMAND ----------

# finding the total number of tracks played and minutes(60000 millisec/min) listened and number of users under each gender
usage_diff = combined_df.groupby('gender').aggregate({'gender':'count', 'ms_played':lambda x: sum(x)/60000, 'track_id':lambda x: len(x.unique()),})
usage_diff.rename(columns={'gender':'Users', 'ms_played':'Minutes_Played', 'track_id':'Tracks_Listened'}, inplace=True)
usage_diff.head(2).plot(kind='bar',figsize=(12, 8),title="Overall Listening Analysis: Female vs Male Users")
# usage_diff.to_csv("https://s3-us-west-1.amazonaws.com/vamsinallabothubucket/MvF_ListeningAnalysis.csv")
usage_diff

# COMMAND ----------

display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Finding the differences in listening contexts of both the Male and Female users. The below graph shows the difference in which male and female users listen to tracks from different contexts

# COMMAND ----------

fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
male_context = combined_df[combined_df['gender']=='male'].groupby(['context']).size().order(ascending=False)
female_context = combined_df[combined_df.gender=='female'].groupby(['context']).size().order(ascending=False)
male_context.plot(kind='pie', ax=axs[0],autopct='%1.1f%%', title='Total tracks played from different contexts: \n Male Users')
female_context.plot(kind='pie', autopct='%1.1f%%', ax=axs[1], title='Total tracks played from different contexts: \n Female Users')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Capturing the number of male and female users in the specified age_range in to male_ageRange and female_ageRange pandas series

# COMMAND ----------

female_ageRange = combined_df.loc[combined_df.gender=='male', 'age_range'].value_counts()
male_ageRange = combined_df.loc[combined_df.gender=='female', 'age_range'].value_counts()

# COMMAND ----------

female_ageRange

# COMMAND ----------


male_ageRange


# COMMAND ----------

# MAGIC %md
# MAGIC * Converting the male_ageRange and female_ageRange series in to mdf and fmdf DataFrames

# COMMAND ----------


mdf = pd.DataFrame(male_ageRange)

fmdf = pd.DataFrame(female_ageRange)

mdf['age_range'] = mdf.index

fmdf['age_range'] = fmdf.index

mdf.reset_index(drop=True, inplace=True)

fmdf.reset_index(drop=True, inplace=True)


# COMMAND ----------

mdf

# COMMAND ----------

fmdf

# COMMAND ----------

# MAGIC %md
# MAGIC * Merging the DataFrames on the age_range value

# COMMAND ----------


ageRange_df = pd.merge(mdf,fmdf, on='age_range')

ageRange_df.columns = ['male_users', 'age_range', 'female_users']

ageRange_df.set_index('age_range', inplace = True)


# COMMAND ----------

# MAGIC %md
# MAGIC * Comparing the age range of Male and Female users listening to the songs on the Spotify

# COMMAND ----------

ageRange_df.plot(kind='pie',figsize=(12,6), autopct='%1.1f%%', title="Age difference of Male and Female Users", subplots = True)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * A function to classify the session based on hour of the day

# COMMAND ----------


def get_session(x):
    end_time = time.localtime(x).tm_hour
    if 0 < end_time <= 6:
        return 'Mid Night'
    elif 6 < end_time <= 12:
        return 'Morning'
    elif 12 < end_time <= 18:
        return 'Noon'
    else:
        return 'Evening'

# COMMAND ----------

# MAGIC %md
# MAGIC * using the get_session function, set the session for each user based on the end_timestamp and add the "session" series to the combined_df DataFrame

# COMMAND ----------

combined_df['session'] = combined_df['end_timestamp'].apply(get_session)
combined_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Plotting the graph showing the difference in number male and female users listening at different sessions

# COMMAND ----------

df = combined_df.groupby(['session','age_range', 'gender']).size()
df.unstack().plot(kind='bar', figsize=(15,16),title="Male Vs Female Users listening at different Sessions")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Plotting the graph showing the difference in number male and female users listening at different sessions

# COMMAND ----------

df.unstack(level=[0,1]).drop(['unknown'], axis=0).plot(kind="bar", figsize=(18, 15), title="Male Vs Female Users listening at different Sessions")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * categorizing the total users based on the session and age_Range for each session. Finding the average amount of stream per each session

# COMMAND ----------

s = combined_df.groupby(['session','age_range']).size()
s_top = s.order(ascending=False).head(10)
s_avg = combined_df.groupby(['session'])['ms_played'].agg({'avg':lambda x: np.mean(x)/60000,'total':lambda x: np.sum(x)/3600000})

# COMMAND ----------

s.head()

# COMMAND ----------

# MAGIC %md
# MAGIC * Total stream in hours per session and the Average time streaming of song in each session

# COMMAND ----------

fig, axs = plt.subplots(ncols=2,figsize=(18,12))
s_avg['total'].plot(kind='bar',title='Total Streams in Hours/Session',ax=axs[0],color='c')
s_avg['avg'].plot(kind='bar',title='Avg Streams in Minutes/Session',ax=axs[1])
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * No of users active per session and number of users of certain age range

# COMMAND ----------

fig, axs = plt.subplots(ncols=2,figsize=(25,9))
combined_df.groupby('age_range')['age_range'].agg({'Total No of Users':np.size}).sort('Total No of Users',ascending=False).plot(kind='bar',ax=axs[0],title="User Demographics: Divided by Age")
s_top.plot(kind='barh',color=['g'],legend=False,ax=axs[1],title=" Active users per Session")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Top 10 countries with least premium subscribers

# COMMAND ----------

subs=combined_df.groupby(['country','product']).size().reset_index()
subs[subs['product']=='premium'].rename(columns={0:'Total Premium Subscriptions'}).sort('Total Premium Subscriptions',ascending=True).set_index(['country']).head(20).plot(kind='bar',figsize=(10,7),title="20 Least Premium Subscribed Countries",stacked=True)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC * Top 10 Countries with more premium subscirbers

# COMMAND ----------

subs[subs['product']=='premium'].rename(columns={0:'Premium Subscribers'}).sort('Premium Subscribers',ascending=False).set_index(['country']).head(10).plot(kind='bar',figsize=(12,7),title="Top 10 Countries with highest Premium Subscribers",color='g', stacked=True)
display()

# COMMAND ----------

# MAGIC %md
# MAGIC Longevity of Premium subscriber accounts in each country

# COMMAND ----------

usr=combined_df.groupby(['user_id','country','age_range','acct_age_weeks'])
premium_usr=usr.aggregate({'product':lambda x: list(set(x))})
premium_usr['Premium'] = premium_usr['product'].apply(lambda x: x.__contains__('premium'))
premium_usr[premium_usr['Premium']==True].reset_index().groupby('country').aggregate({'country':'count','acct_age_weeks':np.mean}).sort(['acct_age_weeks'],ascending=False).rename(columns={'country':'Premium Users','acct_age_weeks':'Avg Account Age'}).plot(kind='bar',figsize=(20,10),title='Average working period of the account held by Premium users')
display()

# COMMAND ----------

premium_usr[premium_usr['Premium'] == False].reset_index().groupby('country').aggregate({'country':'count','acct_age_weeks':np.mean}).sort(['country'],ascending=False).rename(columns={'country':'Non Premium Users','acct_age_weeks':'Avg Account Age'}).plot(kind='bar',figsize=(20,10),title='Avg age of account held by Non premium users of a country')
display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Graph showing number of Premium users and Free users

# COMMAND ----------

fig, axs = plt.subplots(ncols=2, figsize=(15, 6))
combined_df[combined_df['product']=='premium'].groupby(['context']).size().order(ascending=False).plot(kind='pie',ax=axs[0],title='Premium Content Users: Songs Context', autopct='%1.1f%%')
combined_df[combined_df['product']!='premium'].groupby(['context']).size().order(ascending=False).plot(kind='pie',ax=axs[1],title='Free Content Users: Songs Context', autopct='%1.1f%%')
display()

# COMMAND ----------


