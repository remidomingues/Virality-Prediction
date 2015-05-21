Virality Prediction
===================
Overview
--------
This project aims at predicting the virality of tweets and hashtags on Twitter based on a regression model and a classifier trained by a machine learning algorithms.

Technologies
------------
- Python
- MongoDB

Data
----
The data used is composed of random English tweets crawled from the Twitter API during 3 days.

Setup
-----
- Install MongoDB: `sudo apt-get install mongodb`
- Install the Python libs: `sudo pip install -r requirements.txt` If you get trouble installing the module `h5py`, try installing the corresponding package manually first, for example with `sudo apt-get install libhdf5-dev` on Ubuntu.

Run
---
1. Fill your Twitter API key in `authExample.py` and rename the file `auth.py`
2. Execute `python stream.py` to get as many tweets as you want (at least 3 days if you can)
3. Remove the duplicates and add an index on the tweets ID
    - Execute the shell command `mongo`
    - In the mongo shell, execute `db.Tweets.createIndex({id: 1}, {unique: true, dropDups: true})`
4. Execute `python retweetUpdater.py` a few days later to update the number of retweets of each tweet. This timeframe will be the one of your model when predicting the virality of a tweet or hashtag
5. Predict the virality of previously retrieved hashtags: `python viralityPrediction.py`. This step can be replaced by the following ones:
    - Extract the features from the tweets stored in database: `python featuresExtractor.py`
    - Train your regression model or classifier to predict the number of retweets or the tweet virality class based on the features: `python regression.py`
    - Build an inverted index, giving a list of tweets for each hashtag: `python hashtagIndex.py`
    - Predict the virality of hashtags: `python viralityPrediction.py`
6. Predict the virality of new hashtags using current tweets: use `python twitterSearch.py` to retrieve tweets IDs from the Twitter API for a given hashtag. When you have retrieved the features for each tweet, you can feed those features to `viralityPrediction.py` in order to output a retweet count or virality prediction
