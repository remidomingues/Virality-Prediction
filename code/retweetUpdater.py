#Enable remote debugging
import ptvsd
ptvsd.enable_attach(secret = 'paulisdebuggingfromspace', address = ('0.0.0.0', 1243))

from tweepy import OAuthHandler
from tweepy import API
from tweepy import TweepError
from auth import TwitterAuth

import time
import tweepy
import pymongo
import os


# Class to update the retweet count of MongoDB tweet records
class RetweetUpdater:

    # perform authentication
    def __init__(self):
        auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
        auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)
        self.api = API(auth)

    def statusLookup(self, tweetIDs, collection):
        for attempt in range(10):
            try:
                tweets = self.api.statuses_lookup(tweetIDs)
                # extract features of each tweet
                for tweet in tweets:
                    tweetID = tweet.id
                    doc = collection.find_one({"id": tweetID})
                    doc['retweet_count'] = tweet.retweet_count
                    collection.save(doc)
                break
            except TweepError:
                print "tweepy error - try again"

    def updateCount(self, outputDatabaseName, collectionName):
        # open file containing list of tweet-IDs from 2013
        try:
            print "Connecting to database"
            conn=pymongo.MongoClient()
            outputDB = conn[outputDatabaseName]
            collection = outputDB[collectionName]

            apiCalls = 0
            tweetCount = 0
            tweetIDs = []
            docs = []

            print "Start Twitter API calls"
            # Iterate over all documents
            for doc in collection.find(no_cursor_timeout=True):
                tweetID = doc['id']
                if (doc['retweet_count'] > 0) continue;
                doc['retweet_count'] = -1
                collection.save(doc)
                tweetIDs.append(tweetID)
                docs.append(doc)
                tweetCount += 1
                # when 100 tweet IDs have been collected, make API call
                if len(tweetIDs) == 100:
                    self.statusLookup(tweetIDs, collection)
                    tweetIDs = []
                    docs = []
                    apiCalls += 1
                    print("Progress: "+str(tweetCount))
                # after 180 API calls, wait for 15 minutes (Twitter rate limit)
                if apiCalls == 180:
                    print("Sleep for 15 minutes - "+ time.strftime('%X') + "\n")
                    time.sleep(15*60)
                    apiCalls = 0
            
            # Lookup remaining tweets
            if len(tweetIDs) != 0:
                self.statusLookup(tweetIDs, collection)
                print("Progress: "+str(tweetCount))

            # Remove tweets that were deleted / accounts supspended    
            collection.delete_many({'retweet_count': -1})
        except pymongo.errors.ConnectionFailure, e:
            print "Could not connect to MongoDB: %s" % e 


def main():
    updater = RetweetUpdater()
    outputDatabaseName = "Twitter"
    collectionName = "Tweets"
    updater.updateCount(outputDatabaseName, collectionName)
    

if __name__ == "__main__":
    main()

