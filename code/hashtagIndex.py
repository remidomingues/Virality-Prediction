import pymongo
import pickle

class HashtagIndex:

    #Initialize class
    def __init__(self):
        self.name = "../data/hashtag_index"
        self.index = {}
        try:
            self.loadIndex()
        except:
            print "No index file found, generating from database"
            self.generateIndex("Twitter", "Tweets")


    #Generate index and save to .pkl file
    def generateIndex(self, outputDatabaseName, collectionName):
        try:
            print "Connecting to database"
            conn=pymongo.MongoClient()
            outputDB = conn[outputDatabaseName]
            collection = outputDB[collectionName]
            print "Setting up inverted index"
            #Query database for tweets containing hashtags
            test = collection.find({"entities.hashtags.text" : {"$exists": True}})
            for tweet in test:
                for hashtag in tweet["entities"]["hashtags"]:
                    if hashtag["text"] in self.index:
                        #Add tweet id to previously known hashtag
                        self.index[hashtag["text"]].append(tweet["id"])
                    else:
                        #Add tweet id to new hashtag
                        self.index[hashtag["text"]] = [tweet["id"]]
            #Save to .pkl file
            self.saveIndex()
        except:
            print "Error encountered in generateIndex"

    #Save index to file
    def saveIndex(self):
        print "Saving index to file"
        with open(self.name + ".pkl", "wb") as f:
            pickle.dump(self.index, f, pickle.HIGHEST_PROTOCOL)

    #Load index from file
    def loadIndex(self):
        print "Loading hashtags index..."
        with open(self.name + ".pkl", "rb") as f:
            self.index = pickle.load(f)

    #Returns a list of tweet ID's for the given hashtag
    def find(self, hashtag):
        if hashtag in self.index:
            return self.index[hashtag]
        else:
            return []

def main():
    hashtagIndex = HashtagIndex()
    print hashtagIndex.find("nomore") # Should print [592958600357793793L]
    print hashtagIndex.find('OneDirection')

if __name__ == "__main__":
    main()
