from tweepy import OAuthHandler
from tweepy import API
from auth import TwitterAuth
import time
import h5py
import json
import tweepy
import pymongo
import os

# Allow access to raw json data of status objects
@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status

tweepy.models.Status.first_parse = tweepy.models.Status.parse
tweepy.models.Status.parse = parse

# A simple wrapper for the Twitter API using Tweepy
class TwitterCrawler:

    # perform authentication
    def __init__(self):
        auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
        auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)
        self.api = API(auth)

    # search for given query
    def querySearch(self, query):
        tweets = self.api.search(q=query, count = 50, result_type="recent")
        # print features of each tweet
        for tweet in tweets:
            self.printFeatures(tweet)

    def statusLookup(self, tweetIDs, collection):
        tweets = self.api.statuses_lookup(tweetIDs)
        # extract features of each tweet
        for tweet in tweets:
            collection.insert(json.loads(tweet.json))
        
    # print features of a given tweet
    def printFeatures(self, tweet):
        # general tweet information
        print("text: " + tweet.text)
        print("retweeted: " + str(tweet.retweet_count))
        print("favorited: " + str(tweet.favorite_count))
        print("reply: " + str(tweet.in_reply_to_status_id_str != None))
        print("created_at: " + str(tweet.created_at))
        print("language: " + tweet.lang)
        # if tweet has retweeted_status attribute, tweet is a retweet
        if hasattr(tweet, 'retweeted_status'):
            print("is a retweet: True")
        else:
            print("is a retweet: False")
        # entities information (URLs, hashtags, media, etc.)
        print("hashtag: " + str(tweet.entities.get('hashtags') != []))
        print("media: " + str(tweet.entities.get('media') != []))
        print("user mentions: " + str(tweet.entities.get('user_mentions') != []))
        print("urls: " + str(tweet.entities.get('urls') != []))
        # user details
        print("followers: "+str(tweet.user.followers_count))
        print("following: "+str(tweet.user.friends_count))
        print("list appearances: "+str(tweet.user.listed_count))
        print("number of tweets: "+str(tweet.user.statuses_count))
        print("verified: "+str(tweet.user.verified)+"\n")

    def crawl(self, inputFilename, outputDatabaseName, collectionName):
        # open file containing list of tweet-IDs from 2013
        try:
            print "Connecting to database"
            conn=pymongo.MongoClient()
            outputDB = conn[outputDatabaseName]
            collection = outputDB[collectionName]
            print "Start Twitter API calls"
            with open(inputFilename) as f:
                apiCalls = 0
                tweetCount = 0
                tweetIDs = []
                for line in f:
                    # extract and collect tweet ID
                    parts = line.split()
                    tweetID = parts[0]
                    tweetIDs.append(tweetID)
                    tweetCount += 1
                    # when 100 tweet IDs have been collected, make API call
                    if len(tweetIDs) == 100:
                        self.statusLookup(tweetIDs, collection)
                        tweetIDs = []
                        apiCalls += 1
                        print("Progress: "+str(tweetCount))
                    # after 180 API calls, wait for 15 minutes (Twitter rate limit)
                    if apiCalls == 180:
                        print("Sleep for 15 minutes - "+ time.strftime('%X') + "\n")
                        time.sleep(15*60)
                        apiCalls = 0
        except pymongo.errors.ConnectionFailure, e:
            print "Could not connect to MongoDB: %s" % e 


def main():
    wrapper = TwitterCrawler()
    dataDirectory = "../data/20110123/"
    hdfsFilename = "../data/output.hdf5"
    outputDatabaseName = "Twitter"
    collectionName = "Tweets"
    fileNames = os.listdir(dataDirectory)
    # Iterate over all files contained in data directory
    for filename in fileNames:
        inputFilename = dataDirectory+filename
        wrapper.crawl(inputFilename, outputDatabaseName, collectionName)
    

if __name__ == "__main__":
    main()

