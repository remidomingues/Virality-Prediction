from datetime import datetime
import pymongo
import h5py

class FeatureExtractor:
    # Data description
    FEATURE_LABEL = ['followers_count', 'friends_count', 'listed_count', 'statuses_count',
        'favourites_count', 'days_from_account_creation', 'verified', 'hashtags', 'media',
        'user_mentions', 'urls', 'text_len']
    VIRALITY_LABEL = ['retweet_count']

    # Output file path
    HDF5_FILEPATH = "../data/features.hdf5"

    # MongoDB database and table
    TWITTER_DATABASE = "Twitter"
    TWEETS_TABLE = "Tweets"

    @staticmethod
    def __twitterDateToDaysFromNow(twitterString):
        twitterDate = datetime.strptime(twitterString, '%a %b %d %H:%M:%S +0000 %Y')
        return (datetime.now() - twitterDate).days

    # extract features from tweet and append them to the lists
    @staticmethod
    def __getFeatures(tweet, ids, featuresList, viralityList, keepTweetWithoutHashtags):
        features = []
        if 'retweeted_status' in tweet:
            user = tweet['retweeted_status']['user']
        else:
            user = tweet['user']

        features.append(max(user['followers_count'], 0))
        features.append(max(user['friends_count'], 0))
        features.append(max(user['listed_count'], 0))
        features.append(max(user['statuses_count'], 0))
        features.append(max(user['favourites_count'], 0))
        features.append(FeatureExtractor.__twitterDateToDaysFromNow(user['created_at']))
        if user['verified']:
            features.append(1)
        else:
            features.append(0)

        if 'hashtags' in tweet['entities']:
            features.append(len(tweet['entities']['hashtags']))
            if len(tweet['entities']['hashtags']) == 0 and keepTweetWithoutHashtags == False:
                return
        else:
            if keepTweetWithoutHashtags == False:
                return
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
            features.append(min(len(tweet['text']), 140))
        else:
            features.append(0)

        retweet_count = tweet['retweet_count']
        if retweet_count > 3000000:
            return

        virality = []
        virality.append(max(retweet_count, 0))
        virality.append(max(tweet['favorite_count'], 0))
        virality.append(max(tweet['retweet_count'], 0) + max(tweet['favorite_count'], 0))

        ids.append(tweet['id'])
        featuresList.append(features)
        viralityList.append(virality)

    # connect to MongoDB database and get all tweets then extract features for each tweet
    @staticmethod
    def loadFromDB(limit=0, tweets_id=None, keepTweetWithoutHashtags=False):
        ids = []
        featuresList = []
        viralityList = []

        try:
            conn = pymongo.MongoClient()
            db = conn[FeatureExtractor.TWITTER_DATABASE]
            collection = db[FeatureExtractor.TWEETS_TABLE]
            if tweets_id is None:
                for tweet in collection.find(limit=limit):
                    FeatureExtractor.__getFeatures(tweet, ids, featuresList, viralityList, keepTweetWithoutHashtags)
            else:
                for tweet_id in tweets_id:
                    FeatureExtractor.__getFeatures(db[FeatureExtractor.TWEETS_TABLE].find_one({"id": tweet_id}),
                        ids, featuresList, viralityList, keepTweetWithoutHashtags)

        except pymongo.errors.ConnectionFailure, e:
            print "> Could not connect to MongoDB: %s" % e

        return ids, featuresList, viralityList


    @staticmethod
    def load(force=False, keepTweetWithoutHashtags=False):
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
                ids, features, virality = FeatureExtractor.loadFromDB(keepTweetWithoutHashtags=keepTweetWithoutHashtags)
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

        # ID
        dset_ids = output.create_dataset("IDs", data=(ids))
        dset_ids.attrs["Column 0"] = "ID"

        # Features
        dset_features = output.create_dataset("Features", data=(featuresList))
        col_name = 'Column '
        for idx, label in enumerate(FeatureExtractor.FEATURE_LABEL):
            dset_features.attrs[col_name + str(idx)] = label

        # Virality
        dset_virality = output.create_dataset("Virality", data=(viralityList))
        dset_virality.attrs["Column 0"] = FeatureExtractor.VIRALITY_LABEL[0]
        dset_virality.attrs["Column 1"] = "favorite_count"
        dset_virality.attrs["Column 2"] = "combined_count"
        output.close()


if __name__ == "__main__":
    # ids, features, virality = FeatureExtractor.loadFromDB(tweets_id=[592958600357793793L, 592673811239149568L])
    # ids, features, virality = FeatureExtractor.loadFromDB(limit=0)

    # Load features from DB and dump to disk if required
    ids, features, virality = FeatureExtractor.load(force=True)
