import pandas as pd
import numpy as np
import ast
import csv
import sys
import json
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import ShuffleSplit, learning_curve


"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
"""

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#Pomocne funkcije za preprocesiranje
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
				for item in others_temp:
					others.append(item)
				sum_others += 1
		else:
			data["genres"].iloc[index] = "NO_ASSIGNED_GENRE"

		
		sys.stdout.write("\033[F") #back to previous line 
		sys.stdout.write("\033[K") #clear line 
		print((i/end)*100)
		i += 1

	indexEmpty = data[data["genres"] == "NO_ASSIGNED_GENRE"].index
	data.drop(indexEmpty, inplace=True)
	

	print("-"*30)
	print("NUM OF OTHERS: ")
	print(sum_others)
	print("-"*30)
	print(data)
	print(others)

	others_dict = {}
	for other_genre in others:
		if other_genre in others_dict:
			others_dict[other_genre] += 1
		else:
			others_dict[other_genre] = 1

	others_dict_s = pd.Series(others_dict, name="num_of_occ")
	others_dict_s = others_dict_s.sort_values(ascending=False)
	others_dict_s.to_csv("counted_others.csv")

	data.to_pickle('data.pkl')
	
def assign_super_genre(genres: list):
	 
	super_genres_dict = {
		"metal": ["metal", "grindcore", "metalcore", "nu-metalcore", "thrash", "power metal"],
		"punk": ["punk", "anarcho-punk", "cowpunk", "oi"],
		"blues": ["blues", "boogie-woogie", "boogie"],
		"jazz": ["jazz", "swing", "bop", "ragtime", "dixieland", "schlager"],
		"classical": ["orchestra", "orchestral", "classical", "baroque", "chamber", "renaissance", "piano", "romantic", "romanticism", "symphony", "violin", "opera"],
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
		"show tunes": ["show tunes", "broadway", "movie tunes", "hollywood", "disney"],
		"worship": ["worship", "gospel", "ccm", "christian"],
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

	whole_data_w_genres.to_csv("data-genres.csv")

	print("* Finished with adding genres pardner!")




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

def build_model_SMOTE(X_smote, Y_smote):
	smote = SMOTE(random_state=4)
	X, Y = smote.fit_resample(X_smote, Y_smote)

	x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
	smote_pipeline = make_pipeline(RandomForestClassifier(n_estimators=80, criterion='gini', max_features='auto', max_depth=40))
	smote_model = timer(lambda: smote_pipeline.fit(x_train, y_train))
	return x_test, y_test, smote_model


def build_model_ADASYN(X_ada, Y_ada):
	ada = ADASYN(random_state=4)
	X, Y = ada.fit_resample(X_ada, Y_ada)

	x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
	ada_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100, criterion='gini', max_features='auto', max_depth=50))
	ada_model = timer(lambda: ada_pipeline.fit(x_train, y_train))
	return x_test, y_test, ada_model

def build_model_undersampling(X, Y):
	x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.3)
	nearmiss_pipeline = make_pipeline_imb(NearMiss(), RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', max_depth=100))
	nearmiss_model = timer(lambda: nearmiss_pipeline.fit(x_train, y_train))
	return x_test, y_test, nearmiss_model


def print_results(true_value, prediction):
	print("Accuracy: {}".format(accuracy_score(true_value, prediction)))


if __name__ == "__main__":
	pd.options.mode.chained_assignment = None
	data___ = pd.read_csv("data-genres.csv")


	data = remove_outliers(data___, 3)


	print("* Done readin' feller!")

	X = data[
		["danceability", "energy", "speechiness", "acousticness", "instrumentalness",
		 "valence", "tempo", "duration_ms", "liveness", "popularity"]]
	Y = data['genres']


	min_max = MinMaxScaler()
	X_min_max = min_max.fit_transform(X)

	
	#x_test, y_test, model = build_model(X_min_max, Y)
	x_test, y_test, model = build_model_SMOTE(X_min_max, Y)
	#x_test, y_test, model = build_model_ADASYN(X_min_max, Y)
	#x_test, y_test, model = build_model_undersampling(X_min_max, Y)

	prediction = model.predict(x_test)

	print_results(y_test, prediction)


