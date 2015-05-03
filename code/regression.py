from sklearn import linear_model
import numpy as np
import h5py
import pickle

class RegressionModel:
    # HDFS file path
    HDFS_FILEPATH = "../data/output.hdf5"
    # Traing set proportion
    TRAINING_SIZE = 0.8
    # Serialization file
    SERIALIZATION_FILE = "../data/regression_model"

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
        fileObj = h5py.File(RegressionModel.HDFS_FILEPATH, 'r')
        idList = fileObj["IDs"]
        features = fileObj["Features"]
        virality = fileObj["Virality"]
        return [idList, features, virality]

    @staticmethod
    def load_datasets():
        """
        Return the training and testing datasets containing
        the tweets featured followed by the retweet count
        """
        # Import data
        hdfs_data = RegressionModel.__loadDataFromHDFS()
        # Concatenate the arrays into one along the second axis
        data = np.c_[hdfs_data[1], hdfs_data[2][:, 0]]
        # Shuffle data
        np.random.shuffle(data)
        # Split dataset into training and testing sets
        size = int(len(data) * RegressionModel.TRAINING_SIZE)
        training_set = data[:size]
        testing_set = data[len(data)-size:]
        return [training_set, testing_set]

    def train(self, training_set):
        """
        Train the classifier
        """
        print "Training model..."
        # Features

        X_train = training_set[:, :-1]
        # Retweet count
        Y_train = training_set[:, -1]
        # Model training
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(X_train, Y_train)

    def score(self, testing_set):
        """
        Compute benchmarks according to the testing dataset
        """
        # Features
        X_test = testing_set[:, :-1]
        # Retweet count
        Y_test = testing_set[:, -1]
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

    def load(self):
        """
        Load the classifier from a binary file
        """
        try:
            with open(self.SERIALIZATION_FILE + ".pkl", "rb") as f:
                self.clf = pickle.load(f)
            print "Regression model loaded"
            return True
        except:
            print "Could not load regression model"
            return False

    def dump(self):
        """
        Export the classifier in a binary file
        """
        print "Exporting regression model..."
        try:
            with open(self.SERIALIZATION_FILE + ".pkl", "wb") as f:
                pickle.dump(self.clf, f, pickle.HIGHEST_PROTOCOL)
            return True
        except:
            return False


if __name__ == "__main__":
    [training_set, testing_set] = RegressionModel.load_datasets()
    model = RegressionModel()
    if not model.load():
        model.train(training_set)
        model.dump()
    model.score(testing_set)
    print "Prediction samples: ", model.predict([testing_set[0][:-1], testing_set[1][:-1]])
