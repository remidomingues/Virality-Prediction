import pymongo
import pickle

class HashtagIndex:
    INDEX_FILEPATH = "../data/hashtag_index"

    def __init__(self):
        self.index = {}
        try:
            self.loadIndex()
        except:
            print "Index file not found"
            self.generateIndex("Twitter", "Tweets")

    # Generate index and save to .pkl file
    def generateIndex(self, outputDatabaseName, collectionName):
        try:
            conn=pymongo.MongoClient()
            outputDB = conn[outputDatabaseName]
            collection = outputDB[collectionName]
            print "Building inverted index..."
            # Query database for tweets containing hashtags
            test = collection.find({"entities.hashtags.text" : {"$exists": True}})
            for tweet in test:
                for hashtag in tweet["entities"]["hashtags"]:
                    if hashtag["text"] in self.index:
                        # Add tweet id to previously known hashtag
                        self.index[hashtag["text"]].append(tweet["id"])
                    else:
                        # Add tweet id to new hashtag
                        self.index[hashtag["text"]] = [tweet["id"]]
            # Save to .pkl file
            self.saveIndex()
        except:
            print "Could not generate index. Please check your database connection"

    # Save index to file
    def saveIndex(self):
        print "Saving index to file"
        with open(HashtagIndex.INDEX_FILEPATH + ".pkl", "wb") as f:
            pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)

    # Load index from file
    def loadIndex(self):
        print "Loading hashtags index..."
        with open(HashtagIndex.INDEX_FILEPATH + ".pkl", "rb") as f:
            self.index = pickle.load(f)

    # Returns a list of tweet ID's for the given hashtag
    def find(self, hashtag):
        if hashtag in self.index:
            return self.index[hashtag]
        else:
            return []

    def keys(self):
        """
        Return an array of hashtags
        """
        return self.index.keys()

    def values(self):
        """
        Return an array of arrays of tweets ID
        """
        return self.index.values()

    def items(self, sort=False, descending=False, min_values=0):
        """
        Return the index items, possibly sorted by ascending or descending (descending=True) order
        Only the items having more than min_values values are returned
        """
        if min_values > 1:
            result = [(k, v) for (k, v) in self.index.items() if len(v) >= min_values]
        else:
            result = self.index.items()

        if sort:
            result = sorted(result, key=lambda (k, v): len(v), reverse=descending)

        return result

if __name__ == "__main__":
    hashtagIndex = HashtagIndex()
    print hashtagIndex.find("nomore") # Should print [592958600357793793L]
    print hashtagIndex.find('OneDirection')
