import pymongo
import h5py

class FeatureExtractor:
    HDF5_FILEPATH = "../data/features.hdf5"
    TWITTER_DATABASE = "Twitter"
    TWEETS_TABLE = "Tweets"

    # extract features from tweet and append them to the lists
    @staticmethod
    def __getFeatures(tweet, ids, featuresList, viralityList):
        ids.append(tweet['id'])
        features = []
        features.append(tweet['user']['followers_count'])
        features.append(tweet['user']['friends_count'])
        features.append(tweet['user']['listed_count'])
        features.append(tweet['user']['statuses_count'])
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
        if tweet['user']['verified']:
            features.append(1)
        else:
            features.append(0)
        if 'retweeted_status' in tweet:
            features.append(1)
        else:
            features.append(0)
        features.append(len(tweet['text']))
        featuresList.append(features)
        virality = []
        virality.append(tweet['retweet_count'])
        virality.append(tweet['favorite_count'])
        virality.append(tweet['retweet_count'] + tweet['favorite_count'])
        viralityList.append(virality)

    # connect to MongoDB database and get all tweets then extract features for each tweet
    @staticmethod
    def loadFromDB(tweets_id=None):
        ids = []
        featuresList = []
        viralityList = []

        try:
            conn = pymongo.MongoClient()
            db = conn[FeatureExtractor.TWITTER_DATABASE]
            collection = db[FeatureExtractor.TWEETS_TABLE]
            if tweets_id is None:
                for tweet in collection.find():
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
            with h5py.File(FeatureExtractor.HDF5_FILEPATH, 'r') as f:
                ids = fileObj["IDs"]
                features = fileObj["Features"]
                virality = fileObj["Virality"]
                return ids, features, virality
        except:
            if force:
                ids, features, virality = FeatureExtractor.loadFromDB()
                FeatureExtractor.dump(ids, features, virality)
                return ids, features, virality
            print "> Could not load features"
            return None, None, None

    @staticmethod
    def dump(ids, featuresList, viralityList):
        """
        Save the given features in a HDF5 file
        """
        print "Exporting features"
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
    ids, features, virality = FeatureExtractor.load()
    ids, features, virality = FeatureExtractor.loadFromDB([592958600357793793L, 592673811239149568L])
    ids, features, virality = FeatureExtractor.loadFromDB()
    FeatureExtractor.dump(ids, features, virality)
