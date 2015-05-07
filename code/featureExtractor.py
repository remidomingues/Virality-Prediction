import pymongo
import h5py

class FeatureExtractor:
    HDF5_FILEPATH = "../data/features.hdf5"
    TWITTER_DATABASE = "Twitter"
    TWEETS_TABLE = "Tweets"

    # extract features from tweet and append them to the lists
    @staticmethod
    def __getFeatures(tweet, ids, featuresList, viralityList):
        features = []
        if 'retweeted_status' in tweet:
            features.append(max(tweet['retweeted_status']['user']['followers_count'], 0))
            features.append(max(tweet['retweeted_status']['user']['friends_count'], 0))
            features.append(max(tweet['retweeted_status']['user']['listed_count'], 0))
            features.append(max(tweet['retweeted_status']['user']['statuses_count'], 0))
            if tweet['retweeted_status']['user']['verified']:
                features.append(1)
            else:
                features.append(0)
        else:
            features.append(max(tweet['user']['followers_count'], 0))
            features.append(max(tweet['user']['friends_count'], 0))
            features.append(max(tweet['user']['listed_count'], 0))
            features.append(max(tweet['user']['statuses_count'], 0))
            if tweet['user']['verified']:
                features.append(1)
            else:
                features.append(0)
        if 'hashtags' in tweet['entities']:
            features.append(len(tweet['entities']['hashtags']))
        else:
            features.append(0)
        if 'media' in tweet['entities']:
            features.append(len(tweet['entities']['media']))
        else:
            features.append(0)
        if 'user_mentions' in tweet['entities']:
            features.append(len(tweet['entities']['user_mentions']))
        else:
            features.append(0)
        if 'urls' in tweet['entities']:
            features.append(len(tweet['entities']['urls']))
        else:
            features.append(0)
        if 'text' in tweet:
            features.append(len(tweet['text']))
        else:
            features.append(0)

        virality = []
        virality.append(max(tweet['retweet_count'], 0))
        virality.append(max(tweet['favorite_count'], 0))
        virality.append(max(tweet['retweet_count'], 0) + max(tweet['favorite_count'], 0))

        ids.append(tweet['id'])
        featuresList.append(features)
        viralityList.append(virality)

    # connect to MongoDB database and get all tweets then extract features for each tweet
    @staticmethod
    def loadFromDB(limit=0, tweets_id=None):
        ids = []
        featuresList = []
        viralityList = []

        try:
            conn = pymongo.MongoClient()
            db = conn[FeatureExtractor.TWITTER_DATABASE]
            collection = db[FeatureExtractor.TWEETS_TABLE]
            if tweets_id is None:
                for tweet in collection.find(limit=limit):
                    FeatureExtractor.__getFeatures(tweet, ids, featuresList, viralityList)
            else:
                for tweet_id in tweets_id:
                    FeatureExtractor.__getFeatures(db[FeatureExtractor.TWEETS_TABLE].find_one({"id": tweet_id}),
                        ids, featuresList, viralityList)

        except pymongo.errors.ConnectionFailure, e:
            print "> Could not connect to MongoDB: %s" % e

        return ids, featuresList, viralityList


    @staticmethod
    def load(force=False):
        """
        Load data from the specified HDF5 file. Structure is Dataset => Attributes
        IDs => ID
        Features => [followers_count, friends_count, listed_count, statuses_count,
                     hashtags_count, media_count, user_mention_count, url_count,
                     verified_account, is_a_retweet, tweet_length]
        Virality => [retweet_count, favorite_count, combined_count]
        """
        print "Loading features..."
        try:
            f = h5py.File(FeatureExtractor.HDF5_FILEPATH, 'r')
            ids = f["IDs"]
            features = f["Features"]
            virality = f["Virality"]
            print "> {} rows loaded".format(len(ids))
            return ids, features, virality
        except:
            print "> Could not load features"
            if force:
                print "Loading features from database..."
                ids, features, virality = FeatureExtractor.loadFromDB()
                FeatureExtractor.dump(ids, features, virality)
                print "> {} rows loaded".format(len(ids))
                return ids, features, virality
            return None, None, None

    @staticmethod
    def dump(ids, featuresList, viralityList):
        """
        Save the given features in a HDF5 file
        """
        print "Exporting features..."
        output = h5py.File(FeatureExtractor.HDF5_FILEPATH, "w")
        dset_ids = output.create_dataset("IDs", data=(ids))
        dset_ids.attrs["Column 0"] = "ID"
        dset_features = output.create_dataset("Features", data=(featuresList))
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
        dset_features.attrs["Column 10"] = "tweet_length"
        dset_virality = output.create_dataset("Virality", data=(viralityList))
        dset_virality.attrs["Column 0"] = "retweet_count"
        dset_virality.attrs["Column 1"] = "favorite_count"
        dset_virality.attrs["Column 2"] = "combined_count"
        output.close()


if __name__ == "__main__":
    # ids, features, virality = FeatureExtractor.loadFromDB(tweets_id=[592958600357793793L, 592673811239149568L])
    # ids, features, virality = FeatureExtractor.loadFromDB(limit=10000)

    # Load features from DB and dump to disk if required
    ids, features, virality = FeatureExtractor.load(force=True)
