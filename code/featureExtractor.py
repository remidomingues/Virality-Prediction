import pymongo
import h5py

class FeatureExtractor:

    # initialize lists
    def __init__(self):
        self.idList = []
        self.featureList = []
        self.viralityList = []

    # save features to HDF5 file at given path
    def saveToFile(self, path):
        output = h5py.File(path, "w")
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

    # extract features from tweet and append them to the lists
    def getFeatures(self, tweet):
        self.idList.append(tweet['id'])
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
        self.featureList.append(features)
        virality = []
        virality.append(tweet['retweet_count'])
        virality.append(tweet['favorite_count'])
        virality.append(tweet['retweet_count'] + tweet['favorite_count'])
        self.viralityList.append(virality)

    # connect to MongoDB database outputDatabaseName and get all tweets in collectionName collection
    # then extract features for each tweet and save them in HDF5 file at hdfsPath
    def extractFeatures(self, outputDatabaseName, collectionName, hdfsPath):
        try:
            print "Connecting to database"
            conn=pymongo.MongoClient()
            outputDB = conn[outputDatabaseName]
            collection = outputDB[collectionName]
            print "Extracting features"
            for tweet in collection.find():
                self.getFeatures(tweet)
            print "Saving file"
            self.saveToFile(hdfsPath)

        except pymongo.errors.ConnectionFailure, e:
            print "Could not connect to MongoDB: %s" % e 


def main():
    extractor = FeatureExtractor()
    hdfsPath = "../data/features.hdf5"
    outputDatabaseName = "Twitter"
    collectionName = "Tweets"
    extractor.extractFeatures(outputDatabaseName, collectionName, hdfsPath)


if __name__ == "__main__":
    main()