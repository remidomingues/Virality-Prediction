from sklearn import linear_model
from featureExtractor import FeatureExtractor
import matplotlib.pyplot as plt
from dataAnalysis import DataAnalyser
import numpy as np
import pickle
import random

class RegressionModel:
    # Traing set proportion
    TRAINING_SIZE = 0.8
    # Serialization file
    SERIALIZATION_FILE = "../data/regression_model"
    # Plot files
    ERROR_PLOT_FILENAME_REGRESSION = "prediction_error_regression.png"
    ERROR_PLOT_FILENAME_CLASSIFICATION = "prediction_error_classification.png"
    COEF_PLOT_FILENAME_REGRESSION = "coefficients_regression.png"
    COEF_PLOT_FILENAME_CLASSIFICATION = "coefficients_classification.png"

    @staticmethod
    def load_datasets(balance=False, viral_threshold=0):
        """
        Return the training and testing datasets containing
        the tweets featured followed by the retweet count
        """
        # Import data
        _, features, virality = FeatureExtractor.load(force=True)
        print "Building datasets..."
        # Concatenate the arrays into one along the second axis
        data = np.c_[features, np.array(virality)[:, 0]]

        RegressionModel.__dataset_range(data)
        # Duplicate viral tweets to balance the dataset
        if balance:
            data = RegressionModel.__balance_virality(dataset=data, threshold=viral_threshold)
        # Shuffle data
        np.random.shuffle(data)
        # Split dataset into training and testing sets
        size = int(len(data) * RegressionModel.TRAINING_SIZE)

        # why  was the test set having overlap with the traning set earlier.
        training_set = data[:size]
        testing_set = data[size:]

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

        print "Balancing dataset... ({:.2f}% viral tweets, dataset size increased by {:.2f}%)".format(
            (n_virals / float(len(virality)) * 100), 100 * ((len(virality) - n_virals) * 2 / float(len(virality)) - 1))

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

    @staticmethod
    def __dataset_range(data):
        print "Range of feature and virality values:"
        print "> min={}".format(list(np.min(data, axis=0)))
        print "> max={}".format(list(np.max(data, axis=0)))


    def trainRegression(self, training_set, normalize=False, showPlot=False, savePlot=False):
        """
        Train the classifier        """
        print "Training model..."
        # Features
        X_train = training_set[:, :-1]
        print "Training shape"
        print X_train.shape
        # Retweet count
        Y_train = training_set[:, -1]

        # Model training
        self.clf = linear_model.BayesianRidge(normalize=normalize)
        self.clf.fit(X_train, Y_train)
        # Model coefficients
        print "> Regression coefficients: ", list(self.clf.coef_)
        self.plot_coefficients(showPlot, savePlot, 1)

    def trainClassifier(self, training_set, normalize=False, showPlot=False, savePlot=False):
        """
        Train a LR model with the training set data.
        """
        X_train = training_set[:, :-1]
        # Retweet count

        Y_train = training_set[:, -1]
        median = np.median(Y_train)

        print "Training Median"
        print median

        Y = np.zeros_like(Y_train)
        Y[Y_train > median] =1
        print set(Y)

        self.LR = linear_model.LogisticRegression(penalty='l2')
        self.LR.fit(X_train, Y)

        print "> Classifier coefficients: ", list(self.LR.coef_)
        self.plot_coefficients(showPlot, savePlot, 2)

    ''' 
    This function normalises all the features manually 
    x =  (x-mean/ sigma)

    '''

    def normaliseFeats(self, training_set):
        final_training_set = np.copy(training_set)
        meanVec = np.mean(final_training_set, axis = 0)
        stdVec = np.std(final_training_set, axis = 0)

        meanMat = np.tile(meanVec.transpose(), (training_set.shape[0], 1))
        stdMat = np.tile(stdVec.transpose(), (training_set.shape[0], 1))  

        final_training_set = (final_training_set - meanMat) / stdMat

        return final_training_set



    def scoreRegression(self, testing_set, hashtag=None, showPlot=False, savePlot=False):
        """
        Compute benchmarks according to the testing dataset
        """
        print "Regression score:"
        # Features
        X_test = testing_set[:, :-1]
        # Retweet count
        Y_test = testing_set[:, -1]
        # print X_test.shape

        predictions = self.predictRegression(X_test)

        # Mean squared error
        print "> Residual sum of squares: {:.2f}".format(
            np.mean((predictions - Y_test) ** 2))

        print "Success Rate:"
        print  np.sqrt(np.mean((predictions - Y_test) **2))
        # Variance score: 1 is perfect prediction
        print "> Variance score: %.3f" % self.clf.score(X_test, Y_test)
        self.plot_testing_error(Y_test, predictions, hashtag=hashtag, showPlot=showPlot, savePlot=savePlot)



    def scoreClassifier(self, testing_set, showPlot=True, savePlot=False):
        """
        Score according to the LR model.
        """
        X_test = testing_set[:, :-1]
        Y_test = testing_set[:, -1]

        test_median = np.median(Y_test)
        print "Test Median"
        print test_median

        Y = np.zeros_like(Y_test)
        Y[Y_test > test_median] = 1

        Ypreds = self.LR.predict(X_test)

        Y_t = Y == Ypreds
        Y_f = Y != Ypreds

        tp = np.sum(Y[Y_t] == 1)
        tn = np.sum(Y[Y_t] == 0)
        fp = np.sum(Y[Y_f] == 0)
        fn = np.sum(Y[Y_f] == 1)

        t = tp + tn
        f = fp + fn
        tf = t + f
        ratio = t / float(tf)

        print "classifier accuracy: " + str(ratio)

        values = (tp, tn, fp, fn)
        self.plot_positiveNegatives(values, showPlot=showPlot, savePlot=savePlot)


    def predictRegression(self, features):
        """
        Predict the retweet counts for every tweet based on their features
        """
        result = np.array(self.clf.predict(features))

        result[result < 0] = 0
        return result

    def predictClassifier(self, features):
        """
        Predict the retweet counts for every tweet based on their features
        """
        result = np.array(self.LR.predict(features))

        return result


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


    def plot_testing_error(self, expected, predicted, hashtag=None, showPlot=True, savePlot=False):
        # Plot
        if showPlot or savePlot:
            plt.subplot(211)
            plt.axis([0, max(expected), min(predicted), max(expected)])
            plt.xlabel('Expected ' + FeatureExtractor.VIRALITY_LABEL[0])
            plt.ylabel('Predicted ' + FeatureExtractor.VIRALITY_LABEL[0])
            if hashtag is not None:
                plt.title('Prediction score on testing data (#{}, {} tweets)'.format(hashtag, len(expected)))
            else:
                plt.title('Prediction score on testing data ({} tweets)'.format(len(expected)))
            plt.plot(expected, predicted, 'o')

            plt.subplot(212)
            error = abs(expected - predicted)
            plt.axis([0, max(expected), 0, max(error)])
            plt.xlabel('Expected ' + FeatureExtractor.VIRALITY_LABEL[0])
            plt.ylabel('Prediction error')
            plt.plot(expected, error, 'o')

            if savePlot:
                plt.savefig(DataAnalyser.PLOT_DIR + RegressionModel.ERROR_PLOT_FILENAME_REGRESSION, format='png')
            if showPlot:
                plt.show()


    def plot_coefficients(self, showPlot=True, savePlot=False, ctype =1):
        if showPlot or savePlot:
            if ctype == 1:
                y = list(self.clf.coef_)
            elif ctype == 2:
                y = self.LR.coef_.ravel().tolist()
                factor = 100000000
                y = map(lambda x: x*factor, y)

            x = np.arange(len(y))

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(x, y)
            ax.set_xticks(x)
            xtickNames = ax.set_xticklabels(FeatureExtractor.FEATURE_LABEL)
            plt.setp(xtickNames, rotation=45, fontsize=10)
            ax.set_title('Regression model coefficients')

            if savePlot:
                if ctype == 1:
                    plt.savefig(DataAnalyser.PLOT_DIR + RegressionModel.COEF_PLOT_FILENAME_REGRESSION, format='png')
                else:
                    plt.savefig(DataAnalyser.PLOT_DIR + RegressionModel.COEF_PLOT_FILENAME_CLASSIFICATION, format='png')
            if showPlot:
                plt.show()

    def plot_positiveNegatives(self, values, showPlot=True, savePlot=False):
        ind = np.arange(len(values))
        width = 0.35
        p1 = plt.bar(ind, values, width, color='b')
        plt.ylabel('Count')
        plt.xticks(ind+width/2., ('tp', 'tn', 'fp', 'fn') )        

        if savePlot:
            plt.savefig(DataAnalyser.PLOT_DIR + RegressionModel.ERROR_PLOT_FILENAME_CLASSIFICATION, format='png')
        if showPlot:
            plt.show()





if __name__ == "__main__":
    training_set, testing_set = RegressionModel.load_datasets(balance=True, viral_threshold=50000)
    
    model = RegressionModel()

    # Train linear regression
    model.trainRegression(training_set, normalize=True, showPlot=True, savePlot=True)
    model.scoreRegression(testing_set, showPlot=True, savePlot=True)

    #  Normalise all the features individually before training classifier
    #  this way, the weights of all the features will be on equal par 
    #   and scale will not have any influence 

    training_set = model.normaliseFeats(training_set)
    testing_set = model.normaliseFeats(testing_set)

    # Train logistic regression classifier
    print "\n"
    model.trainClassifier(training_set, normalize= True, showPlot=True, savePlot=True)
    model.scoreClassifier(testing_set, showPlot=True, savePlot=True)

