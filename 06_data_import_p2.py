# Importing Data into Python (Part 2)
# -----------------------------------
import pandas as pd
from urllib.request import urlretrieve


# 1. Importing data from the Internet..
# -------------------------------------
# url: uniform/universal resource locator -> references to web resources
# http: hypertext transfer protocol

# 2 packages: urllib & requests

# download file from web, save it & load into dataframe..
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
# save & load as 'winequality-white.csv'..
urlretrieve(url, 'winequality-white.csv')
df = pd.read_csv('winequality-white.csv', sep=";")
print(df.head())
# .. or load file directly to dataframe..
df = pd.read_csv(url, sep=";")

# download excel files from the web..
url = "http://s3.amazonaws.com/assets.datacamp.com/course/importing_data_into_r/latitude.xls"
xl = pd.read_excel(url, sheet_name=None)
# print sheetnames & head of 1st sheet..
print(xl.keys())
print(xl['1700'].head())

# get request using urllib..
from urllib.request import urlopen, Request
url = "https://www.wikipedia.org/"
request = Request(url)
response = urlopen(request)
html = response.read()
response.close()
# .. use request package for this!

# alternatively, perform http requests using requests package (higher-level library)..
import requests
url = "https://www.wikipedia.org/"
r = requests.get(url)
# html as string..
text = r.text

# scraping the web in python..
# ---
# structured data: has pre-defined data model or organized in a defined manner
# unstructured data: neither of these properties
# .. beautifulsoup package: parse & extract structured data from html
from bs4 import BeautifulSoup
import requests
url = 'https://www.crummy.com/software/BeautifulSoup/'
r = requests.get(url)
html_doc = r.text
soup = BeautifulSoup(html_doc)
# structured..
print(soup.prettify())
# title of webpage..
ttl = soup.title
# text of webpage..
txt = soup.get_text()

# find & extract all urls of the hyperlinks from a webpage..
a_tags = soup.find_all('a') # hyperlinks defined as <a>, but passed to find_all() with no angle brackets
for link in a_tags:
    print(link.get('href'))


# 2. Interacting with APIs to import data from the web..
# ------------------------------------------------------
# ..introduction to APIs & JSONs
# JSON: JavaScript Object Notation (advantage: human readable, unlike pickle-files)
# .. much of the data you get via API's are pacakged as JSONs

# JSONs..
# ---
# loading jsons in python..
import json
with open('data/exa2.json', 'r') as json_file:
    json_data = json.load(json_file)

# print key-value pairs..
for key, value in json_data.items():
    print(key + ':', value)
# alternatively..
for k in json_data.keys():
    print(k + ': ', json_data[k])

# exploring json..
json_data['widget']


# APIs..
# ---
# What's an API?
# - set of protocols & routines
# - bunch of code that allows 2 software programs to communicate with each other

import requests

# get data from the OMDB API..
url = 'http://www.omdbapi.com/?apikey=72bc447a&t=the+social+network'
r = requests.get(url)
print(r.text)

json_data = r.json()
for key, value in json_data.items():
    print(key + ':', value)
# explanation:
# http://www.omdb.api.com: querying the OMDB API
# ?t=hackers: query string, return data for a movie with title (t) 'Hackers'

# checking out wikipedia API..
# .. extract info from wiki for pizza..
url = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&titles=pizza"
r = requests.get(url)
json_data = r.json()
pizza_extract = json_data['query']['pages']['24768']['extract']
print(pizza_extract)


# 3. Diving deep into the Twitter API..
# -------------------------------------
import tweepy

# Store OAuth authentication credentials in relevant variables
access_token = "1092294848-aHN7DcRP9B4VMTQIhwqOYiB14YkW92fFO8k8EPy"
access_token_secret = "X4dHmhPfaksHcQ7SCbmZa2oYBBVSD2g8uIHXsp5CTaksx"
consumer_key = "nZ6EA0FxZ293SxGNg8g8aP0HM"
consumer_secret = "fJGEodwe3KiKUnsYJC3VRndj7jevVvXbK2D5EiJ2nehafRgA6i"

# Pass OAuth details to tweepy's OAuth handler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# .. not continued because it requires a class called 'MyStreamListener' that needs to be created (preview below)..
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet_list = status._json
        self.file.write(json.dumps(tweet_list) + '\n')
        tweet_list.append(status)
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False
        self.file.close()


l = MyStreamListener()
# create your stream object with authentication..
stream = tweepy.Stream(auth, l)
# filter twitter streams to capture data by the keywords..
stream.filter(track=['clinton','trump','sanders','cruz'])










