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
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "key", "mode"]]
    y = data['genres'].str.lower()
    #generic_genres = [ "blues", "r&b", "classical", "country", "dance", "house", "hip hop", "indie",
    # "jazz", "pop", "dub", "adult standards", "chill", "rock", "punk", "metal", "broadway", "latin", "reggae", "rap", "disco"]

    generic_genres = ["blues", "banda", "electro", "house", "dance", "swing" , "orchestra", "cumbia" , "dub", "gospel", "bebop", "bolero", "hardcore", "chanson" ,"adult standards" , "broadway" , "chill" , "classical" , "comedy" , "country", "disco" , "edm" , "emo" , "folk" , "funk" , "hip hop" , "indie" , "jazz" , "latin" , "metal" , "pop" , "punk" , "r&b" , "rap" , "reggae" , "rock" , "soul" , "soundtrack" , "worship"]
    print(y)
    for index, specific_genre in y.items():
        found = False
        for generic_genre_word in generic_genres:
            if generic_genre_word in specific_genre:
                #print("\n" + specific_genre)
                #print("--->" + generic_genre_word)
                y[index] = generic_genre_word
                found = True
                #print(y[index] + "\n")
        if not found:    
            y[index] = "other"
    generic_genres.append("other")
    
    print(y)
    hist_x = []
    hist_y =[]
    for generic_genre in generic_genres:
        hist_x.append(generic_genre)
        hist_y.append(y.eq(generic_genre).sum())
        print(generic_genre)
        print(y.eq(generic_genre).sum())
        print("\n")
    # We can set the number of bins with the `bins` kwarg
    print(hist_y)
    plt.hist(hist_y, bins=len(hist_y))
    #plt.show()
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


#ova funkcija je koristena samo da bi izlistali zanrove koji se najcesce pojavljuju u "other" kako bi ih mogli dodati u generalne zarove
def count_occurances(true_df, generalized_df):
    occurrences = {}
    for index, generalized_genre in generalized_df.items():
        if generalized_genre == "other":
            genre_in_true = true_df[index]
            count = 0
            for index, specific_genre in true_df.items():
                if genre_in_true in specific_genre:
                    count = count + 1
            occurrences[genre_in_true] = count
    
    df_occ = pd.DataFrame(occurrences, index=[0])
    df_occ = df_occ.T
    df_occ = df_occ.rename(columns={0: 'col1'})
    df_occ = df_occ.sort_values(by='col1')
    return df_occ

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    data = pd.read_pickle('data.pkl')
    #print(data)
    print(len(data))
    print(len(data['genres'].unique().tolist()))
    x_train, y_train = read_data(data)
    print(x_train)
    print(y_train.nunique())
    print(len(x_train))
    print(len(y_train))
    
