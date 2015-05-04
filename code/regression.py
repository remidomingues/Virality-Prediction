from sklearn import linear_model
from featureExtractor import FeatureExtractor
import numpy as np
import pickle
import random

class RegressionModel:
    # Traing set proportion
    TRAINING_SIZE = 0.8
    # Serialization file
    SERIALIZATION_FILE = "../data/regression_model"

    @staticmethod
    def load_datasets(balance=False, viral_threshold=50):
        """
        Return the training and testing datasets containing
        the tweets featured followed by the retweet count
        """
        # Import data
        # SWITCH THE FOLLOWING LINES TO BYPASS THE FEATURE EXTRACTOR BUG ===============
        _, features, virality = FeatureExtractor.load(force=True)
        # _, features, virality = FeatureExtractor.loadFromDB()
        # ==============================================================================
        print "Building datasets..."
        # Concatenate the arrays into one along the second axis
        data = np.c_[features, [vir[0] for vir in virality]]
        # Duplicate viral tweets to balance the dataset
        if balance:
            data = RegressionModel.__balance_virality(dataset=data, threshold=viral_threshold)
        # Shuffle data
        np.random.shuffle(data)
        # Split dataset into training and testing sets
        size = int(len(data) * RegressionModel.TRAINING_SIZE)
        training_set = data[:size]
        testing_set = data[len(data)-size:]
        return training_set, testing_set

    @staticmethod
    def __balance_virality(dataset, threshold):
        """
        Increase the weight of viral tweets in the dataset by
        duplicating them so that viral and non viral tweets
        are equally represented
        """
        indexes = []
        new_tweets = []
        virality = dataset[:, -1] >= threshold
        n_virals = virality.sum()

        if n_virals == 0:
            return dataset

        # Number of times a viral tweet must be duplicated
        n_repeat = max(int((len(virality) - n_virals) / n_virals) - 1, 0)
        # To obtain a perfect balance, mod tweets must be duplicated once more
        mod = len(virality) % n_virals

        if n_repeat == 0 and mod == 0:
            return dataset

        print "Balancing dataset... ({} virals / {} total: {:.2f}% viral)".format(
            n_virals, len(virality), (n_virals / float(len(virality) * 100)))

        # Insert n_repeat times each viral tweet in the new_tweets list
        for idx, value in enumerate(virality):
            if value:
                indexes.append(idx)
                for _ in xrange(n_repeat):
                    new_tweets.append(dataset[idx])

        # Duplicate mod random tweets once more
        random.shuffle(indexes)
        for i in xrange(mod):
            new_tweets.append(dataset[indexes[i]])

        return np.vstack((dataset, new_tweets))

    def train(self, training_set, normalize=False):
        """
        Train the classifier
        """
        print "Training model..."
        # Features
        X_train = training_set[:, :-1]
        # Retweet count
        Y_train = training_set[:, -1]
        # Model training
        self.clf = linear_model.Lasso(normalize=normalize)

        print np.amin(Y_train)
        print np.amin(X_train)
        print np.max(Y_train)
        print np.max(X_train)
        print X_train[np.argmin(X_train)/11]

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
        print "Loading regression model..."
        try:
            with open(self.SERIALIZATION_FILE + ".pkl", "rb") as f:
                self.clf = pickle.load(f)
            return True
        except:
            print "> Could not load regression model"
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
    training_set, testing_set = RegressionModel.load_datasets(balance=True, viral_threshold=50)
    model = RegressionModel()
    if not model.load():
        model.train(training_set, normalize=True)
        model.dump()
    model.score(testing_set)
    print "Prediction samples: ", model.predict([testing_set[0][:-1], testing_set[1][:-1]])
