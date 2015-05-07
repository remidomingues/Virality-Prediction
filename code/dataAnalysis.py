import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from featureExtractor import FeatureExtractor

class DataAnalyser:

    FEATURE_LABEL = ['followers_count', 'friends_count', 'listed_count', 'statuses_count', 'verified', 'hashtags', 'media', 'user_mentions', 'urls', 'text_len']
    VIRALITY_LABEL = ['retweet_count']

    @staticmethod
    def load_datasets(balance=False, viral_threshold=0):
        """
        Return the datasets containing the tweets featured followed by the retweet count
        """
        # Import data
        _, features, virality = FeatureExtractor.load(force=True, keepTweetWithoutHashtags=True)
        print "Building datasets..."
        # Concatenate the arrays into one along the second axis
        data = np.c_[features, np.array(virality)[:, 0]]
        return pd.DataFrame(data, columns=(DataAnalyser.FEATURE_LABEL+DataAnalyser.VIRALITY_LABEL))

    @staticmethod
    def describeData(df):
    	print "Dataframe statistics:"
        print df.describe()

    @staticmethod
    def plotRetweetInfluence(df):
        print "Retweet influence:"
        df.plot(x=DataAnalyser.FEATURE_LABEL[0], y=DataAnalyser.VIRALITY_LABEL[0], kind='scatter')


if __name__ == "__main__":
    df = DataAnalyser.load_datasets(balance=False)
    DataAnalyser.describeData(df)
    DataAnalyser.plotRetweetInfluence(df)
    plt.show()

