import numpy as np
import math

class NormaliseFeatures:
    RETWEET_COUNT_THRESH = 5
    FAV_COUNT_THRESH = 2
    COMBINED_COUNT_THRESH = 10

# THE FOLLOWING METHOD MAY BE REPLACED BY normalize=true WHEN TRAINING THE REGRSSION MODEL
    # @staticmethod
  #   def normaliseCount(self):
		# self.idList, self.features, self.virality = NormaliseFeatures.__loadDataFromHDFS()
		# cardinality = self.virality.shape[0]
		# virality = np.asarray(self.virality)
		# print cardinality

		# retweet_inds = []
		# fav_inds = []
		# comb_inds  = []

		# for i in range(cardinality):
		# 	if virality[i,0] > NormaliseFeatures.RETWEET_COUNT_THRESH:
		# 		retweet_inds.append(i)

		# 	if virality[ i, 1] > NormaliseFeatures.FAV_COUNT_THRESH:
		# 		fav_inds.append(i)

		# 	if virality[i, 2] > NormaliseFeatures.COMBINED_COUNT_THRESH:
		# 		comb_inds.append(i)


		# # print len(retweet_inds)
		# viral_inds = list(set(retweet_inds) | set(fav_inds))
		# viral_count = len(viral_inds)
		# factor = int(math.ceil((cardinality - viral_count) / viral_count))
		# print factor

		# imp_tweet_count = virality[ np.where( virality > NormaliseFeatures.RETWEET_COUNT_THRESH)]
		# return self._duplicateRows(viral_inds, factor)

if __name__ == '__main__':
	nf = NormaliseFeatures()
	idList, featureList, virality = nf.normaliseCount()