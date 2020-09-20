import pandas as pd
import numpy as np
import ast
import csv
import sys
import json
import time

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, learning_curve


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

	
def read_genres_(data):
	end = len(data.index)
	sum_others = 0
	i = 1
	others = []
	others_temp = []

	for index, row in data.iterrows():
		genre_list = eval(row["genres"])
		if len(genre_list) > 0:
			assigned_genre, others_temp = assign_super_genre(genre_list)
			#TALK
			if data['speechiness'].iloc[index] > 0.5:
				data['genres'].iloc[index] = "talk"
			else:
				data['genres'].iloc[index] = assigned_genre

			if assigned_genre == "other":
				others.append(others_temp)
				sum_others += 1
		else:
		    data["genres"].iloc[index] = "NO_ASSIGNED_GENRE"
		
		sys.stdout.write("\033[F") #back to previous line 
		sys.stdout.write("\033[K") #clear line 
		print((i/end)*100)
		i += 1

	indexEmpty = data[data["genres"] == "NO_ASSIGNED_GENRE"].index
	data.drop(indexEmpty, inplace=True)
	

	#data = deal_with_others(data)

	#print("-"*30)
	#print("NUM OF OTHERS: ")
	#print(sum_others)
	#print("-"*30)
	#print(data)
	#print(others)

	data.to_pickle('data.pkl')

def remove_outliers(data: pd.DataFrame, threshold):
		
	genres = data["genres"].unique()
	print(genres)


	for genre in genres:
		z_scores = pd.DataFrame()
		print("*" * 51)
		print("*\tDoing for {}".format(genre))
		print("*" * 51)
		acousticness = data.acousticness[data["genres"] == genre]
		danceability = data.danceability[data["genres"] == genre]
		duration_ms = data.duration_ms[data["genres"] == genre]
		energy = data.energy[data["genres"] == genre]
		instrumentalness = data.instrumentalness[data["genres"] == genre]
		liveness = data.liveness[data["genres"] == genre]
		loudness = data.loudness[data["genres"] == genre]
		speechiness = data.speechiness[data["genres"] == genre]
		tempo = data.tempo[data["genres"] == genre]
		valence = data.valence[data["genres"] == genre]
		popularity = data.popularity[data["genres"] == genre]
		key = data.key[data["genres"] == genre]

		#calculate z-scores for every parameter
		data["acousticness_zs"] = (acousticness - acousticness.mean())/acousticness.std()
		data["danceability_zs"] = (danceability - danceability.mean())/danceability.std()
		data["duration_ms_zs"] = (duration_ms - duration_ms.mean())/duration_ms.std()
		data["energy_zs"] = (energy - energy.mean())/energy.std()
		data["instrumentalness_zs"] = (instrumentalness - instrumentalness.mean())/instrumentalness.std()
		data["liveness_zs"] = (liveness - liveness.mean())/liveness.std()
		data["loudness_zs"] = (loudness - loudness.mean())/loudness.std()
		data["speechiness_zs"] = (speechiness - speechiness.mean())/speechiness.std()
		data["tempo_zs"] = (tempo - tempo.mean())/tempo.std()
		data["valence_zs"] = (valence - valence.mean())/valence.std()
		data["popularity_zs"] = (popularity - popularity.mean())/popularity.std()
		data["key_zs"] = (key - key.mean())/key.std()

		print(data.size)
		#filter the data with every entry being labeled as outlier if it is outside of 3rd standard deviation
		index_outlier = data[
			(data["acousticness_zs"] < -1 * threshold) | (data["acousticness_zs"] > threshold) |
			(data["danceability_zs"] < -1 * threshold) | (data["danceability_zs"] > threshold) |
			(data["duration_ms_zs"] < -1 * threshold) | (data["duration_ms_zs"] > threshold) |
			(data["energy_zs"] < -1 * threshold) | (data["energy_zs"] > threshold) |
			(data["instrumentalness_zs"] < -1 * threshold) | (data["instrumentalness_zs"] > threshold) |
			(data["liveness_zs"] < -1 * threshold) | (data["liveness_zs"] > threshold) |
			(data["loudness_zs"] < -1 * threshold) | (data["loudness_zs"] > threshold) |
			(data["speechiness_zs"] < -1 * threshold) | (data["speechiness_zs"] > threshold) |
			(data["tempo_zs"] < -1 * threshold) | (data["tempo_zs"] > threshold) |
			(data["valence_zs"] < -1 * threshold) | (data["valence_zs"] > threshold) |
			(data["popularity_zs"] < -1 * threshold) | (data["popularity_zs"] > threshold) |
			(data["key_zs"] < -1 * threshold) | (data["key_zs"] > threshold)].index
			
		data.drop(index_outlier, inplace=True)
		print(data.size)
	return data

			

def assign_super_genre(genres: list):
	 
	super_genres_dict = {
		"metal": ["metal", "grindcore", "metalcore", "nu-metalcore", "thrash", "power metal"],
		"punk": ["punk", "anarcho-punk", "cowpunk", "oi"],
		"blues": ["blues", "boogie-woogie", "boogie"],
		"jazz": ["jazz", "swing", "bop", "ragtime", "dixieland", "schlager"],
		"classical": ["orchestra", "orchestral", "classical", "baroque", "chamber", "renaissance", "piano", "romantic", "romanticism", "symphony", "violin"],
		"latino": ["latino", "merengue", "mariachi", "reggaeton", "cumbia", "salsa", "cha-cha-cha", "tango", "bachata", "mexican", "samba", "merenge", "flamenco", "rumba", "cuban", "latin"],
		"r&b/soul": ["r&b", "soul", "disco", "doo-wop", "funk", "motown", "rhythm and blues"],
		"country": ["country", "bluegrass", "honky tonk", "honky tonk", "americana", "redneck"],
		"reggae": ["reggae", "roots reggae", "dub", "lovers rock"],
		"hip-hop/rap": ["rap", "hip hop", "bounce"],
		"edm": ["breakbeat", "dance", "grime", "dubstep", "electronic", "eurodance", "eurobeat", "house", "edm", "dnb", "drum and bass", "jungle", "techno", "trance", "chillhop", "lo-fi"],
		"folk": ["folk", "polka"],
		"rock": ["rock", "post-grunge", "grunge"],
		"pop": ["britpop", "pop"],
		"adult standards": ["adult standards"],
		"talk": ["comedy", "poetry", "reading"],
	}

	super_genres_dict_scores = {}

	for super_genre in super_genres_dict:
		super_genres_dict_scores[super_genre] = 0
		for sub_genre in super_genres_dict[super_genre]:
			for genre in genres:
				if sub_genre in genre:
					super_genres_dict_scores[super_genre] += 1

	most_relevant_genre = max(super_genres_dict_scores, key=super_genres_dict_scores.get)

	if super_genres_dict_scores[most_relevant_genre] != 0:
		return most_relevant_genre, []
	else:
		return "other", genres

def find_genre_by_artist(artist, genre_df):
	"""
	for index_artist, value_artist in genre_df.iterrows():
			if genre_df.loc[index_artist, "artists"] == artist:
				return data.loc[index_artist, "genres"]
	"""
	artist_index = genre_df[genre_df['artists'].str.contains(artist, regex=False)].index 
	if genre_df.loc[artist_index, "genres"].size != 0:
		return genre_df.loc[artist_index, "genres"].iloc[0]
	else:
		return "NO_ASSIGNED_GENRE"

def add_generic_genres_to_data(data_w_generes, whole_data_wo_generes):
	whole_data_w_genres = whole_data_wo_generes

	end = len(whole_data_w_genres.index)
	i = 1
	for index_entry, value_entry in whole_data_w_genres.iterrows():
		artists_str = whole_data_w_genres.loc[index_entry, "artists"]
		artist_list = eval(artists_str)
		whole_data_w_genres.loc[index_entry, "genres"] = find_genre_by_artist(artist_list[0], data_w_generes)
		sys.stdout.write("\033[F") #back to previous line 
		sys.stdout.write("\033[K") #clear line 
		print((i/end)*100)
		i += 1

	indexEmpty = whole_data_w_genres[whole_data_w_genres["genres"] == "NO_ASSIGNED_GENRE"].index
	whole_data_w_genres.drop(indexEmpty, inplace=True)

	whole_data_w_genres.to_csv("data_w_added_genres_1.csv")

	print("* Finished with adding genres pardner!")

def timer(f):
	start = time.time()
	res = f()
	end = time.time()
	print("Fitting took: {}".format(end - start))
	return res

def build_model(X, Y):
	x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
	pipeline = make_pipeline(RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', max_depth=100))
	model = timer(lambda: pipeline.fit(x_train, y_train))
	return x_test, y_test, model

if __name__ == "__main__":
	pd.options.mode.chained_assignment = None
	#read_genres()
	#data__ = pd.read_csv("data_w_genres.csv")
	#read_genres_(data__)
	#data_w_genres = pd.read_pickle('data.pkl')
	#whole_data_wo_genres = pd.read_csv('data.csv')
	#data = add_generic_genres_to_data(data_w_genres, whole_data_wo_genres) 
	data__ = pd.read_csv("data_w_added_genres_1.csv")
	#read_genres_(dataa)
	#data = pd.read_pickle("data.pkl")



	data = remove_outliers(data_, 3)


	print("* Done readin' feller!")

	X = data[
		["danceability", "energy", "speechiness", "acousticness", "instrumentalness",
		 "valence", "tempo", "duration_ms", "liveness", "popularity"]]
	Y = data['genres']
	"""
	#HEATMAP
	fig = plt.figure(figsize=(13,6))
	labels = X.columns.tolist()
	ax1 = fig.add_subplot(111)
	ax1.set_xticklabels(labels,rotation=90, fontsize=10)
	ax1.set_yticklabels(labels,fontsize=10)
	plt.imshow(X.corr(), cmap='bwr', interpolation='nearest')
	plt.colorbar()
	ax1.set_xticks(np.arange(len(labels)))
	ax1.set_yticks(np.arange(len(labels)))
	plt.show()

	plt.scatter(X["energy"], X["danceability"])
	plt.show()
	"""
	min_max = MinMaxScaler()
	X_min_max = min_max.fit_transform(X)
	x_test, y_test, model = build_model(X_min_max, Y)

	plot_learning_curve()

	prediction = model.predict(x_test)
	print("Accuracy: {}".format(accuracy_score(y_test, prediction)))

	
	prediction = model.predict(X)
	print("Training set accuracy: {}".format(accuracy_score(Y, prediction)))



	"""
	data_test = pd.read_csv("data.csv")
	
	X_data_test = data_test[
		["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
		 "valence", "tempo", "liveness", "popularity", "key", "mode"]]
	data_test["genres"] = model.predict(X_data_test)

	data_test.drop(["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
		 "valence", "tempo", "liveness", "popularity", "key", "mode"], axis=1)
	data_test.to_csv("predicted_data.csv")

	"""









































	#print("* Startin' classification")
	#data = pd.read_pickle("data.pkl")
	#print(len(data['genres'].unique().tolist()))
	
	

	#print(data[(data.genres == "rock")].mean())
	#plt.scatter(data["energy"], data["loudness"])

	#data_ = data[(data["genres"].str.contains("funk")) & (~data["genres"].str.contains("rock")) & (~data["genres"].str.contains("metal")) & (~data["genres"].str.contains("breaks"))]
	#data_ = data[(data["genres"].str.contains("house")) ^ (data["genres"].str.contains("techno")) ^ (data["genres"].str.contains("trance")) ^ (data["genres"].str.contains("dubstep")) ^ (data["genres"].str.contains("drum and bass")) ^ (data["genres"].str.contains("dnb")) ^ (data["genres"].str.contains("hardstyle"))]
	#data_ = data[data["genres"].str.contains("disco")]
	#data_ = data[(data["speechiness"] > 0.4) & (data["liveness"] > 0.1)]

	#print(data_)
	#print(data_.describe())

	#for i in range(1, 14):
		#plt.hist(data_.iloc[:, i])
		#plt.show()
	

	""" 
  
	
	#x_train, y_train = dana_genre(data)
	#x_train = normalize(x_train)
	print("* Done splittin'")

	
  
	#HEATMAP
	fig = plt.figure(figsize=(13,6))
	labels = x_train.columns.tolist()
	ax1 = fig.add_subplot(111)
	ax1.set_xticklabels(labels,rotation=90, fontsize=10)
	ax1.set_yticklabels(labels,fontsize=10)
	plt.imshow(x_train.corr(), cmap='bwr', interpolation='nearest')
	plt.colorbar()
	ax1.set_xticks(np.arange(len(labels)))
	ax1.set_yticks(np.arange(len(labels)))
	plt.show()
	

	"""

  

	"""
	#REGRESSION
	sel = SelectFromModel(LogisticRegression(penalty='l1', C=100))
	sel.fit(x_train, y_train)
	print(sel.get_support())

	x_train = sel.transform(x_train)
	x_test = sel.transform(x_test)


	sc = StandardScaler()
	x_train = sc.fit_transform(x_train)
	x_test = sc.transform(x_test)


	"""
	

	#PCA
	#x_train, x_test = do_PCA(4, x_train, x_test)
	#print("* Done PCA-in'")

	#classifier =  SVC(kernel='linear',gamma=0.01, C=0.01)
	#classifier = GradientBoostingClassifier(max_depth=3, n_estimators=700)
	#classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', max_depth=100)
	#classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=4)
	#classifier = KNeighborsClassifier(n_neighbors=1)
	
	
   
	"""
	classifier.fit(x_train, y_train)
	y_pred = classifier.predict(x_test)
	acc = accuracy_score(y_test, y_pred)
	print(acc)
	
	#title = "Learning curve"
	#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
	#fig, axes = plt.subplots(3, 2, figsize=(10, 15))
	#plot_learning_curve(classifier, title, x_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

	#plt.show()


	x_ext, other_ext = read_ext_data("data.csv")

	x_ext = sc.transform(x_ext)

	y_pred_ext = classifier.predict(x_ext)

	other_ext.insert(loc=3, column="predicted_genres", value=y_pred_ext)

	other_ext.to_csv("predicted_data.csv")
	print(other_ext.nunique())

	
	"""
	
