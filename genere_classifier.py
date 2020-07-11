import pandas as pd
import numpy as np
import ast
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC


def read_data(data: pd.DataFrame):
    x = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "key", "mode"]]
    y = data['genres'].str.lower()
    #generic_genres = [ "blues", "r&b", "classical", "country", "dance", "house", "hip hop", "indie",
    # "jazz", "pop", "dub", "adult standards", "chill", "rock", "punk", "metal", "broadway", "latin", "reggae", "rap", "disco"]

    generic_genres = ["blues", "banda", "electro", "house", "dance", "swing" , "orchestra", "cumbia" , "dub", "gospel", "bebop", "bolero", "hardcore", "chanson" ,"adult standards" , "broadway" , "chill" , "classical" , "comedy" , "country", "disco" , "edm" , "emo" , "folk" , "funk" , "hip hop" , "indie" , "jazz" , "latin" , "metal" , "pop" , "punk" , "r&b" , "rap" , "reggae" , "rock" , "soul" , "soundtrack" , "worship"]

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
    
    hist_x = []
    hist_y =[]
    for generic_genre in generic_genres:
        hist_x.append(generic_genre)
        hist_y.append(y.eq(generic_genre).sum())
        #print(generic_genre)
        #print(y.eq(generic_genre).sum())
        #print("\n")
    # We can set the number of bins with the `bins` kwarg
    print(hist_y)
    plt.hist(hist_y, bins=len(hist_y))
    #plt.show()
    return x, y


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

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.4)
    print(y_train.nunique())
    print(y_test.nunique())

    print(len(x_train))
    print(len(x_test))

    
    """ 
    pca = PCA(n_components=3)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    """
    #classifier =  SVC(kernel='linear',gamma=0.01, C=0.01)
    classifier = RandomForestClassifier(n_estimators=300, criterion='gini', max_features='auto', max_depth=200, random_state=7)
    #classifier = KNeighborsClassifier(n_neighbors=300, weights='uniform', algorithm='auto')
    #classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=4)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)