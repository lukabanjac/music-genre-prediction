import pandas as pd

data = pd.read_csv("data_w_added_genres_1.csv")

genres = data["genres"].unique()
print(genres)


for genre in genres:
	z_scores = pd.DataFrame()
	print("*" * 111)
	print("*\tDoing for {}".format(genre))
	print("*" * 111)
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
		(data["acousticness_zs"] < -3) | (data["acousticness_zs"] > 3) |
		(data["danceability_zs"] < -3) | (data["danceability_zs"] > 3) |
		(data["duration_ms_zs"] < -3) | (data["duration_ms_zs"] > 3) |
		(data["energy_zs"] < -3) | (data["energy_zs"] > 3) |
		(data["instrumentalness_zs"] < -3) | (data["instrumentalness_zs"] > 3) |
		(data["liveness_zs"] < -3) | (data["liveness_zs"] > 3) |
		(data["loudness_zs"] < -3) | (data["loudness_zs"] > 3) |
		(data["speechiness_zs"] < -3) | (data["speechiness_zs"] > 3) |
		(data["tempo_zs"] < -3) | (data["tempo_zs"] > 3) |
		(data["valence_zs"] < -3) | (data["valence_zs"] > 3) |
		(data["popularity_zs"] < -3) | (data["popularity_zs"] > 3) |
		(data["key_zs"] < -3) | (data["key_zs"] > 3)].index
		
	data.drop(index_outlier, inplace=True)
	

	indexEmpty = data[(data["genres"] == "NO_ASSIGNED_GENRE") | (data["genres"] == "NO_PARAMETER_FIT")].index
	#data.drop(indexEmpty, inplace=True)

	print(data.size)
