import numpy as np
import math
import h5py
import pickle

class NormaliseFeatures:
    # HDFS file path
    HDFS_FILEPATH = "../data/output.hdf5"
    HDFS_NORM_FILEPATH = "../data/normfeatures.hdf5"

    RETWEET_COUNT_THRESH = 5
    FAV_COUNT_THRESH = 2
    COMBINED_COUNT_THRESH = 10


    @staticmethod
    def __loadDataFromHDFS():
        """
        Load data from the specified HDFS file. Structure is Dataset => Attributes
        IDs => ID
        Features => [followers_count, friends_count, listed_count, statuses_count,
                     hashtags_count, media_count, user_mention_count, url_count,
                     verified_account, is_a_retweet, tweet_length]
        Virality => [retweet_count, favorite_count, combined_count]
        """
        print "Importing data..."
        fileObj = h5py.File(NormaliseFeatures.HDFS_FILEPATH, 'r')
        idList = fileObj["IDs"]
        features = fileObj["Features"]
        virality = fileObj["Virality"]
        return idList, features, virality

    @classmethod
    def _duplicateRows(self, indices, count):
        '''
        This function makes new positive tweets; each tweet is
        repeated by (false-true)/true number, so that the number of 
        positive and negative tweets is equal.

        '''
    	# count = 57
        newIdList = np.copy(self.idList)
        newIdList = newIdList[:, np.newaxis]
        newFeatures = np.copy(self.features)
        newVirality = np.copy(self.virality)

        for i in indices:
        	feat = self.features[i];
        	ID = self.idList[i]
        	vir = self.virality[i]
        	featArr = np.tile(feat, ( count, 1))
        	virArr = np.tile(vir, (count, 1))
        	IDArr = np.tile(ID, (count, 1))
        	newFeatures = np.vstack((newFeatures, featArr))
        	newVirality = np.vstack((newVirality, virArr))
        	newIdList = np.vstack((newIdList, IDArr))

        	# for j in range(count):
        	# 	newFeatures = np.vstack((newFeatures, feat))
        	# 	newVirality = np.vstack((newVirality, vir))
        		# np.vstack((newIdList, ID))

        return newIdList, newFeatures, newVirality


    @classmethod
    def writeToHDFS(self, idList, featureList, viralityList):
        '''
            rewriting new normalised data to new hdfs file 

        '''

        print 'Writing to a new hdfs file .....'
        output = h5py.File(NormaliseFeatures.HDFS_NORM_FILEPATH, "w")
        dset_ids = output.create_dataset("IDs", data=(idList))
        dset_ids.attrs["Column 0"] = "ID"
        dset_features = output.create_dataset("Features", data=(featureList))
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
        print 'Done Writing ....'



    @classmethod
    def normaliseCount(self):
		self.idList, self.features, self.virality = NormaliseFeatures.__loadDataFromHDFS()
		cardinality = self.virality.shape[0]
		virality = np.asarray(self.virality)
		print cardinality

		retweet_inds = []
		fav_inds = []
		comb_inds  = []

		for i in range(cardinality):
			if virality[i,0] > NormaliseFeatures.RETWEET_COUNT_THRESH:
				retweet_inds.append(i)

			if virality[ i, 1] > NormaliseFeatures.FAV_COUNT_THRESH:
				fav_inds.append(i)

			if virality[i, 2] > NormaliseFeatures.COMBINED_COUNT_THRESH:
				comb_inds.append(i)

		
		# print len(retweet_inds)
		viral_inds = list(set(retweet_inds) | set(fav_inds))
		viral_count = len(viral_inds)
		factor = int(math.ceil((cardinality - viral_count) / viral_count))
		print factor

		imp_tweet_count = virality[ np.where( virality > NormaliseFeatures.RETWEET_COUNT_THRESH)]
		return self._duplicateRows(viral_inds, factor)
	



if __name__ == '__main__':
	nf = NormaliseFeatures()
	idList, featureList, virality = nf.normaliseCount()
    	nf.writeToHDFS(idList, featureList, virality)

    


