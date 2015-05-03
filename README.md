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
The data used is the Twitter 2011 TREC dataset (http://trec.nist.gov/data/tweets).

Setup
-----
- Install MongoDB: `sudo apt-get install mongodb`
- Install the Python libs: `sudo pip install -r requirements.txt`
If you get trouble installing the module `h5py`, try installing the corresponding package manually first, for example with `sudo apt-get install libhdf5-dev` on Ubuntu.

Run
---
- Fill your Twitter API key in `authExample.py` and rename the file `auth.py`
- [TODO: Run Python scripts to get the data]
- Add an index on the tweets ID:
    - `mongo`
    - `db['Tweets'].createIndex({id: 1}, {unique: true})`
- TODO [update db]
- Train your regression model: `python regression.py`