import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from featureExtractor import FeatureExtractor

class DataAnalyser:
    PLOT_DIR = "../plots/"

    @staticmethod
    def load_datasets(balance=False, viral_threshold=0):
        """
        Return the datasets containing the tweets featured followed by the retweet count
        """
        # Import data
        _, features, virality = FeatureExtractor.load(force=True, keepTweetWithoutHashtags=False)
        print "Building datasets..."
        # Concatenate the arrays into one along the second axis
        data = np.c_[features, np.array(virality)[:, 0]]
        return pd.DataFrame(data, columns=(FeatureExtractor.FEATURE_LABEL+FeatureExtractor.VIRALITY_LABEL))

    @staticmethod
    def describeData(df):
    	print "Dataframe statistics:"
        print df.describe()

    @staticmethod
    def plotRetweetInfluence(df):
        print "Plot retweet influence:"
        for feature in FeatureExtractor.FEATURE_LABEL:
            print "Feature: " + feature
            df.plot(x=feature, y=FeatureExtractor.VIRALITY_LABEL[0], kind='scatter')
            plt.savefig(DataAnalyser.PLOT_DIR+feature+".png", format='png')
            plt.clf()  # Clear the figure for the next loop



if __name__ == "__main__":
    df = DataAnalyser.load_datasets(balance=False)
    DataAnalyser.describeData(df)
    DataAnalyser.plotRetweetInfluence(df)

