from regression import RegressionModel
from hashtagIndex import HashtagIndex
from featureExtractor import FeatureExtractor
from dataAnalysis import DataAnalyser
import matplotlib.pyplot as plt
import numpy as np
import math
import operator

class ViralityPrediction:
    SCORE_PLOT_FILENAME = "hashtags_score.png"
    # If CLASSIFICATION is set to True, classification is used, otherwise regression
    CLASSIFICATION = True
    # K defines the number of results in the Top-K virality predictions
    K = 10

    def __init__(self, normalize=False, balance=False, tweet_threshold=0, score=False, dump_model=True):
        """
        Import or train the regression model
        """
        self.model = RegressionModel()
        if not self.model.load():
            training_set, testing_set = RegressionModel.load_datasets(
                balance=balance, viral_threshold=tweet_threshold)

            if ViralityPrediction.CLASSIFICATION == True:
                training_set = self.model.normaliseFeats(training_set)
                testing_set = self.model.normaliseFeats(testing_set)
                self.model.trainClassifier(training_set, normalize=normalize)
                if score:
                    self.model.scoreClassifier(testing_set)

            else:
                self.model.trainRegression(training_set, normalize=normalize)
                if score:
                    self.model.scoreRegression(testing_set)

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
            if ViralityPrediction.CLASSIFICATION:
                tweets_values = self.model.predictClassifier(value)
            else:
                tweets_values = self.model.predictRegression(value)
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

    def score(self, expected, predicted, labels=None, showPlot=True, savePlot=False):
        if showPlot or savePlot:
            x = np.arange(len(expected))
            width = 0.8
            ticks = x + x * width

            fig = plt.figure()
            ax = fig.add_subplot(111)
            bar1 = ax.bar(ticks, expected, color='green')
            bar2 = ax.bar(ticks + width, predicted, color='blue')
            ax.set_xlim(-width, (ticks + width)[-1] + 2 * width)
            ax.set_ylim(0, max(max(expected), max(predicted)) * 1.05)
            ax.set_xticks(ticks + width)
            if labels is not None:
                xtickNames = ax.set_xticklabels(labels)
                plt.setp(xtickNames, rotation=45, fontsize=10)
            ax.set_title('Expected and predicted retweet count per hashtag')
            ax.legend((bar1[0], bar2[0]), ('Expected', 'Predicted'))

            if savePlot:
                plt.savefig(DataAnalyser.PLOT_DIR + ViralityPrediction.SCORE_PLOT_FILENAME, format='png')
            if showPlot:
                plt.show()

        return np.mean(predicted - expected) ** 2


if __name__ == "__main__":
    vp = ViralityPrediction(normalize=True, balance=True, tweet_threshold=50000,
        score=False, dump_model=False)
    hashtagIndex = HashtagIndex()

    virality = {}
    hashtags_features = {}
    hashtags_virality = {}
    hashtags = [k for (k, v) in hashtagIndex.items(sort=True, descending=True, min_values=100)]
    print "Extracting features..."
    for hashtag in hashtags:
        _, featureList, vir = FeatureExtractor.loadFromDB(tweets_id=hashtagIndex.find(hashtag))
        hashtags_features[hashtag] = featureList
        hashtags_virality[hashtag] = vir
        virality[hashtag] = sum(np.array(vir)[:, 0])

    # Sort predictions by value and print top-K results
    predictions = vp.predict(hashtags_features)
    sorted_predictions = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)
    print "\nTop " + str(ViralityPrediction.K) + " virality predictions:"
    for i in range(0, min(ViralityPrediction.K, len(sorted_predictions))):
        print sorted_predictions[i]

    # Sort expected virality by value and print top-K results
    #sorted_virality = sorted(virality.items(), key=operator.itemgetter(1), reverse=True)
    #print "\nTop " + str(ViralityPrediction.K) + " virality expectations:"
    #for i in range(0, min(ViralityPrediction.K, len(sorted_virality))):
    #    print sorted_virality[i]
    