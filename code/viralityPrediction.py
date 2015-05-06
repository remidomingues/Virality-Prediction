from regression import RegressionModel
from hashtagIndex import HashtagIndex
from featureExtractor import FeatureExtractor
import numpy as np
import math

class ViralityPrediction:
    def __init__(self, normalize=False, balance=False, tweet_threshold=0, dump_model=True):
        """
        Import or train the regression model
        """
        self.model = RegressionModel()
        if not self.model.load():
            training_set, _ = RegressionModel.load_datasets(
                balance=balance, viral_threshold=tweet_threshold)
            self.model.train(training_set, normalize=normalize)
            if dump_model:
                self.model.dump()

    def predict(self, hashtags, hashtag_threshold=None):
        """
        Return a dictionary containing for each hashtag its virality prediction
        - hashtags: dictionary hashtag -> array of tweets features
        - hashtag_threshold: if defined, the dictionary value will be a boolean
            set at true if the hashtag goes viral, false otherwise.
            If not defined, the value will be the number of retweets
        Features are [followers_count, friends_count, listed_count,
            statuses_count, hashtags_count, media_count, user_mention_count,
            url_count, verified_account, is_a_retweet, tweet_length]
        """
        values = {}
        for key, value in hashtags.iteritems():
            tweets_values = self.model.predict(value)
            hashtag_value = sum(tweets_values)
            if hashtag_threshold is not None:
                if hashtag_value >= hashtag_threshold:
                    values[key] = 1
                else:
                    values[key] = 0
            else:
                # Round to the nearest 10 below the current value
                values[key] = max(0, int(math.floor(hashtag_value / 10.0)) * 10)

        return values

    def score(self, X, Y):
         return np.mean(X - Y) ** 2


if __name__ == "__main__":
    vp = ViralityPrediction(normalize=True, balance=False, dump_model=False)
    hashtagIndex = HashtagIndex()

    virality = {}
    hashtags_features = {}
    hashtags = ['OneDirection', 'news', 'bigbang', 'nowplaying']
    print "Extracting features..."
    for hashtag in hashtags:
        _, featureList, vir = FeatureExtractor.loadFromDB(tweets_id=hashtagIndex.find(hashtag))
        hashtags_features[hashtag] = featureList
        virality[hashtag] = sum(np.array(vir)[:, 0])

    print "Predicted hashtags virality:"
    result = vp.predict(hashtags_features)
    print result
    # print vp.predict(hashtags_features, hashtag_threshold=50)
    print "Expected hashtags virality:"
    print virality
    print "Residual sum of squares: {:.2f}".format(vp.score(
        np.array([result[h] for h in hashtags]),
        np.array([virality[h] for h in hashtags])))
