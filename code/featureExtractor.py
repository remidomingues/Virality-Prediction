from tweepy import OAuthHandler
from tweepy import API
from auth import TwitterAuth
import time
import h5py

# A simple wrapper for the Twitter API using Tweepy
class FeatureExtractor:

    # perform authentication
    def __init__(self):
        auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
        auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)
        self.api = API(auth)
        self.idList = []
        self.featureList = []
        self.viralityList = []

    def saveToFile(self, filename):
        output = h5py.File(filename, "w")
        dset_ids = output.create_dataset("IDs", data=(self.idList))
        dset_ids.attrs["Column 0"] = "ID"
        dset_features = output.create_dataset("Features", data=(self.featureList))
        dset_features.attrs["Column 0"] = "followers_count"
        dset_features.attrs["Column 1"] = "friends_count"
        dset_features.attrs["Column 2"] = "listed_count"
        dset_features.attrs["Column 3"] = "statuses_count"
        dset_features.attrs["Column 4"] = "hashtags_count"
        dset_features.attrs["Column 5"] = "media_count"
        dset_features.attrs["Column 6"] = "user_mention_count"
        dset_features.attrs["Column 7"] = "url_count"
        dset_features.attrs["Column 8"] = "verified_account"
        dset_features.attrs["Column 9"] = "is_a_retweet"
        dset_virality = output.create_dataset("Virality", data=(self.viralityList))
        dset_virality.attrs["Column 0"] = "retweet_count"
        dset_virality.attrs["Column 1"] = "favorite_count"
        dset_virality.attrs["Column 2"] = "combined_count"
        output.close()

    # search for given query
    def querySearch(self, query):
        tweets = self.api.search(q=query, count = 50, result_type="recent")
        # print features of each tweet
        for tweet in tweets:
            self.getFeatures(tweet)
            self.printFeatures(tweet)

    def statusLookup(self, tweetIDs):
        tweets = self.api.statuses_lookup(tweetIDs)
        # print features of each tweet
        for tweet in tweets:
            self.getFeatures(tweet)
            #self.printFeatures(tweet)
            
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

    def getFeatures(self, tweet):
        self.idList.append(tweet.id)
        features = []
        features.append(tweet.user.followers_count)
        features.append(tweet.user.friends_count)
        features.append(tweet.user.listed_count)
        features.append(tweet.user.statuses_count)
        if tweet.entities.get('hashtags') != None:
            features.append(len(tweet.entities.get('hashtags')))
        else:
            features.append(0)
        if tweet.entities.get('media') != None:
            features.append(len(tweet.entities.get('media')))
        else:
            features.append(0)
        if tweet.entities.get('user_mentions') != None:
            features.append(len(tweet.entities.get('user_mentions')))
        else:
            features.append(0)
        if tweet.entities.get('urls') != None:
            features.append(len(tweet.entities.get('urls')))
        else:
            features.append(0)
        if tweet.user.verified:
            features.append(1)
        else:
            features.append(0)
        if hasattr(tweet, 'retweeted_status'):
            features.append(1)
        else:
            features.append(0)
        self.featureList.append(features)
        virality = []
        virality.append(tweet.retweet_count)
        virality.append(tweet.favorite_count)
        virality.append(tweet.retweet_count + tweet.favorite_count)
        self.viralityList.append(virality)

    def extractFeaturesToFile(self, inputFilename, outputFilename):
        # open file containing list of tweet-IDs from 2013
        with open(inputFilename) as f:
            apiCalls = 0
            tweetCount = 0
            tweetIDs = []
            for line in f:
                # extract and collect tweet ID
                parts = line.split()
                tweetID = parts[2]
                tweetIDs.append(tweetID)
                tweetCount += 1
                # when 100 tweet IDs have been collected, make API call
                if len(tweetIDs) == 100:
                    self.statusLookup(tweetIDs)
                    tweetIDs = []
                    apiCalls += 1
                    print("progress: "+str(tweetCount))
                # after 180 API calls, wait for 15 minutes (Twitter rate limit)
                if apiCalls == 180:
                    print("sleep for 15 minutes - "+ time.strftime('%X') + "\n")
                    time.sleep(15*60)
                    apiCalls = 0

        # save extracted features to file
        self.saveToFile(outputFilename)


def main():
    wrapper = FeatureExtractor()
    inputFilename = "../data/qrels.microblog2013.txt"
    outputFilename = "../data/output.hdf5"
    wrapper.extractFeaturesToFile(inputFilename, outputFilename)


if __name__ == "__main__":
    main()

