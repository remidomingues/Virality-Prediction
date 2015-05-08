from tweepy import OAuthHandler
from tweepy import API
from auth import TwitterAuth
import datetime

# A simple wrapper for the Twitter Search API using Tweepy
class TwitterSearch:

    @staticmethod
    # perform authentication
    def __authenticate():
        auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
        auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)
        return API(auth)

    @staticmethod
    # search for given query
    def querySearch(query, maxAge=24, requests=180, lang="en"):
        """ 
        query specifies the query string to search for,
        maxAge specifies maximum age of the last returned tweet in hours,
        requests specifies the maximum number of requests (each request returns 100 results),
        lang specifies the language of the tweets to search for. 
        """
        api = TwitterSearch.__authenticate()
        requests = min(requests, 180)
        tweetIDs = []
        now = datetime.datetime.now()
        earlier = now - datetime.timedelta(hours = maxAge)
        oldestID = 0
        for i in range(0,requests):
            if oldestID == 0:
                tweets = api.search(q=query, count = 100, lang=lang, result_type="recent")
            else:
                tweets = api.search(q=query, count = 100, lang=lang, result_type="recent", max_id = oldestID-1)
            if len(tweets) == 0:
                return tweetIDs
            oldestID = tweets[len(tweets)-1].id
            for tweet in tweets:
                if tweet.created_at < earlier:
                    return tweetIDs
                tweetIDs.append(tweet.id)
        return tweetIDs

def main():
    tweetIDs = TwitterSearch.querySearch("stockholm")
    
if __name__ == "__main__":
    main()