Virality Prediction
===================
Overview
--------
This project aims at predicting the virality of a hashtag on Twitter based on a regression model trained by a machine learning algorithm.

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
- Install the Python libs: `sudo pip install -r requirements.txt`
If you get trouble installing the module `h5py`, try installing the corresponding package manually first, for example with `sudo apt-get install libhdf5-dev` on Ubuntu.

Run
---
- Fill your Twitter API key in `authExample.py` and rename the file `auth.py`
- Execute `python stream.py` to get as many tweets as you want
- Add an index on the tweets ID:
    - `mongo`
    - `db['Tweets'].createIndex({id: 1}, {unique: true})`
- Execute `python retweetUpdated.py` a few days later to update the number of retweets of each tweet
- Extract the feature from the tweets stored in database: `python featuresExtractor.py`
- Train your regression model to predict the number of retweets based on the features: `python regression.py`
- Build an inverted index, giving a list of tweets for each hashtag: `python hashtagIndex.py`
- Predict the virality of a hashtag: `python viralityPrediction.py`