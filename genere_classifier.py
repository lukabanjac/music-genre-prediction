import pandas as pd
import numpy as np
import ast
from sklearn import preprocessing, mixture
from sklearn.metrics import v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt


def read_data(data: pd.DataFrame):
    x = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness", "liveness",
         "valence"]]
    y = data['genres'].str.lower()
    generic_genres = [ "blues", "r&b", "classical", "country", "electronic", "hip hop", "new wave", "jazz", "pop", "rock", "punk", "metal" "spoken"]
    print(y)
    for index, specific_genre in y.items():
        for generic_genre_word in generic_genres:
            if generic_genre_word in specific_genre:
                print(specific_genre)
                y[index] = generic_genre_word
                break
            else:
                y[index] = "other"
                break
        print(y[index])
    return x, y

"""     Breakbeat.
Chiptune.
Downtempo.
Drum and bass.
Dub.
Dubstep.
Electro.
Electronica. """

def read_data__():
    #za_poslije_jer_sam_sacuvao_pickle
    #funkcija koja uzima samo prvi zanr od liste zanrova, i eliminise one podatke koji nemaju ni jedan zanr; ukupno oko 27623, kada se pobrise ima 18091
    #ukupno 1784 zanra koja treba svesti na jedan od 12
    data = pd.read_csv("data_w_genres.csv", encoding='utf-8')
    data = data.drop(columns=['artists'])
    #x = data.drop(['genres', 'artists', 'duration_ms', 'liveness'], axis=1)
    for index, value in data['genres'].items():
        list_ = ast.literal_eval(value)
        if len(list_) > 0:
            data['genres'][index] = list_[0]
        else:
            data = data.drop(index=index)
    data.to_pickle('data.pkl')
    print(data)

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    data = pd.read_pickle('data.pkl')
    #print(data)
    print(len(data))
    print(len(data['genres'].unique().tolist()))
    read_data(data)







    """ ss = data['genres'].str.contains('a cappella')
    dd = data['instrumentalness']
    i=0
    for x in ss:
        i = i + 1
        if x == True:
            print(data['genres'][i])
    i = 0
    for x in dd:
        i = i + 1
        if x > 0.8:
            print(data['genres'][i]) """
    #print(data.to_string().encode('utf-8'))