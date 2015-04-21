from tweepy import OAuthHandler
from tweepy import API
from auth import TwitterAuth

# A simple wrapper for the Twitter API using Tweepy
class FeatureExtractor:

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

def main():
    wrapper = FeatureExtractor()
    wrapper.querySearch("#SthlmDL")

if __name__ == "__main__":
    main()

