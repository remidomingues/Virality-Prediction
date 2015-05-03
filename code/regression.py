from sklearn import linear_model
import numpy as np
import h5py

class RegressionModel:
    # HDFS file path
    HDFS_FILEPATH = "../data/features.hdf5"
    # Traing set proportion
    TRAINING_SIZE = 0.8

    def __init__(self):
        self.load_dataset()

    def loadDataFromHDFS(self):
        """
        Load data from the specified HDFS file. Structure is Dataset => Attributes
        IDs => ID
        Features => [followers_count, friends_count, listed_count, statuses_count,
                     hashtags_count, media_count, user_mention_count, url_count,
                     verified_account, is_a_retweet]
        Virality => [retweet_count, favorite_count, combined_count]
        """
        print "Importing data..."
        fileObj = h5py.File(self.HDFS_FILEPATH, 'r')
        idList = fileObj["IDs"]
        features = fileObj["Features"]
        self.feature_size = len(features)
        virality = fileObj["Virality"]
        self.virality_size = len(virality)
        return [idList, features, virality]

    def load_dataset(self):
        # Import data
        hdfs_data = self.loadDataFromHDFS()
        # Concatenate the three arrays into one along the second axis
        data = np.c_[hdfs_data[0], hdfs_data[1], hdfs_data[2]]
        print data[0]
        # Shuffle data
        np.random.shuffle(data)
        # Split dataset into training and testing sets
        size = int(len(data) * self.TRAINING_SIZE)
        self.training_set = data[:size]
        self.testing_set = data[len(data)-size:]

    def train(self):
        print "Training model..."
        # Features
        X_train = self.training_set[:, 1:self.feature_size+1]
        # Retweet count
        Y_train = self.training_set[:, self.feature_size+1]
        # Model training
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(X_train, Y_train)
        # Compute model efficiency
        self.score()

    def score(self):
        # Features
        X_test = self.testing_set[:, 1:11]
        # Retweet count
        Y_test = self.testing_set[:, 11]
        # Model coefficients
        print "Coefficients: ", self.clf.coef_
        # Mean squared error
        print "Residual sum of squares: {:.2f}".format(
            np.mean((self.clf.predict(X_test) - Y_test) ** 2))
        # Variance score: 1 is perfect prediction
        print "Variance score: %.2f" % self.clf.score(X_test, Y_test)

    def predict(self, features):
        """
        Predict the retweet counts for every tweet based on their features
        """
        return self.clf.predict(features)


if __name__ == "__main__":
    model = RegressionModel()
    model.train()
    features = [[950, 1585, 34, 3173, 0, 0, 0, 2, 0, 0],
                [23988, 1770, 164, 123556, 1, 0, 0, 1, 0, 0]]
    print "Prediction samples: ", model.predict(features)
