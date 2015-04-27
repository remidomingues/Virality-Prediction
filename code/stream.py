from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from auth import TwitterAuth
import json
import pymongo

#Very simple (non-production) Twitter stream example
#1. Download / install python and tweepy (pip install tweepy)
#2. Fill in information in auth.py
#3. Run as: python streaming_simple.py
#4. It will keep running until the user presses ctrl+c to exit
#All output stored to output.json (one tweet  per line)

class StdOutListener(StreamListener):

	def __init__(self, outputDatabaseName, collectionName):
		try:
			print "Connecting to database"
			conn=pymongo.MongoClient()
			outputDB = conn[outputDatabaseName]
			self.collection = outputDB[collectionName]
			self.counter = 0
		except pymongo.errors.ConnectionFailure, e:
			print "Could not connect to MongoDB: %s" % e 
	
	#This function gets called every time a new tweet is received on the stream
	def on_data(self, data):	
		#Convert the data to a json object (shouldn't do this in production; might slow down and miss tweets)
		datajson=json.loads(data)

		#Check the language
		if "lang" in datajson and datajson["lang"] == "en" and "text" in datajson:
			self.collection.insert(datajson)

			#See Twitter reference for what fields are included -- https://dev.twitter.com/docs/platform-objects/tweets
			text=datajson["text"] #The text of the tweet
			self.counter += 1
			print(str(self.counter) + " " +text) #Print it out

	def on_error(self, status):
		print("ERROR")
		print(status)

	def on_connect(self):
		print("You're connected to the streaming server.")

if __name__ == '__main__':
	try:
		# Database settings
		outputDatabaseName = "Twitter"
		collectionName = "Tweets"

		#Create the listener
		l = StdOutListener(outputDatabaseName, collectionName)
		auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
		auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)

		#Connect to the Twitter stream
		stream = Stream(auth, l)	

		stopWords = ["a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours	ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"]

		#Terms to track
		stream.filter(track=stopWords)

	except KeyboardInterrupt:
		#User pressed ctrl+c -- get ready to exit the program
		pass
