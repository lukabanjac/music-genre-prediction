import pandas as pd
import numpy as np
import ast
import csv
import sys
import json

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import ShuffleSplit, learning_curve


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def read_data(data: pd.DataFrame):
    x = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "liveness"]]
    y = data['genres'].str.lower()
    #generic_genres = [ "blues", "r&b", "classical", "country", "dance", "house", "hip hop", "indie",
    # "jazz", "pop", "dub", "adult standards", "chill", "rock", "punk", "metal", "broadway", "latin", "reggae", "rap", "disco"]

    generic_genres = ["blues", "banda", "electro", "house", "dance", "swing" , "orchestra", "cumbia" , "dub", "gospel", "bebop", "bolero", "hardcore", "chanson" ,"adult standards" , "broadway" , "chill" , "classical" , "comedy" , "country", "disco" , "edm" , "emo" , "folk" , "funk" , "hip hop" , "indie" , "jazz" , "metal" , "pop" , "punk" , "r&b" , "rap" , "reggae" , "rock" , "soul" , "psy", "soundtrack" , "worship"]

    
    df_occ = pd.read_pickle('occ.pkl')
    df_occ_striped = df_occ[df_occ['col1'] > 50]
    df_occ_striped_list = df_occ_striped.index.tolist()

    for item in df_occ_striped_list:
        generic_genres.append(item)

    for index, specific_genre in y.items():
        found = False
        for generic_genre_word in generic_genres:
            if not found:
                if generic_genre_word in specific_genre:
                    #print("\n" + specific_genre)
                    #print("--->" + generic_genre_word)
                    if generic_genre_word == "psy":
                        print(specific_genre)
                        print("\n")
                    y[index] = generic_genre_word
                    found = True
                #print(y[index] + "\n")
        if not found:    
            y[index] = "other"
 
    return x, y

    """ 
    new_x = x
    new_x.insert(11, "genres", y)
    print(new_x)
    new_x_q = new_x[new_x['genres'] != 'other']

    
    x = new_x_q[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "key", "mode", "popularity"]]
    y = new_x_q['genres'].str.lower()

    #generic_genres.append("other")
    
    hist_x = []
    hist_y =[]
    for generic_genre in generic_genres:
        hist_x.append(generic_genre)
        print(generic_genre)
        print(y.eq(generic_genre).sum())
        hist_y.append(y.eq(generic_genre).sum())
    
    plt.scatter(x['danceability'], x['energy'], s=1, alpha=0.5)
    plt.show()
    plt.scatter(x['acousticness'], x['speechiness'], s=1, alpha=0.5)
    plt.show()
    plt.scatter(x['energy'], x['mode'], s=1, alpha=0.5)
    plt.show()
    plt.scatter(x['danceability'], x['tempo'], s=1, alpha=0.5)
    plt.show() 
    """
    



def read_genres():
    #za_poslije_jer_sam_sacuvao_pickle
    #funkcija koja uzima samo prvi zanr od liste zanrova, i eliminise one podatke koji nemaju ni jedan zanr; ukupno oko 27623, kada se pobrise ima 18091
    #ukupno 1784 zanra koja treba svesti na jedan od 12

    
    data = pd.read_csv("data_w_genres.csv", encoding='utf-8')
    #x = data.drop(['genres', 'artists', 'duration_ms', 'liveness'], axis=1)
    end = len(data.index)
    sum_others = 0
    i = 1
    for index, value in data['genres'].items(): #uzmi samo zanrove
        genre_list = ast.literal_eval(value) #parsiraj u listu
        if len(genre_list) > 0:
            most_relevant_genre = find_most_relevant_genre(genre_list)
            if most_relevant_genre != "": #posto most_relevant_genre vraca samo prazan string ako nije nasao
                data['genres'][index] = most_relevant_genre #znaci ako jeste nadjen, ubacimo u skup
            else: #ako je vracen prazan string od most_relevant_genre, odnosno nije nadjen ni jedan zanr, izbacujemo tu pjesmu iz skupa
                data['genres'][index] = "other"
                sum_others += 1
        else: #ako pjesma nema ni jedan zanr, izbaci
            data = data.drop(index=index)
        
        sys.stdout.write("\033[F") #back to previous line 
        sys.stdout.write("\033[K") #clear line 
        print((i/end)*100)
        i += 1
    print("NUM OF OTHERS: ")
    print(sum_others)
    print("-"*12)
    #print(word_dict)
    #with open('test.csv', 'w') as f:
    #    w = csv.DictWriter(f, word_dict.keys())
    #    w.writeheader()
    #    w.writerow(word_dict)

    data.to_pickle('data.pkl')


#najrelevantiniji zanr za odredjenu pjesmu
#prolazi kroz listu svih zanrova i gleda koliko puta se pojavljuje jedan od generickih zanrova u zanrovima koji su opsti za ovu pjesmu, potom uzima max i vraca najrelevantniji zanr iz skupa generickih
def find_most_relevant_genre(genres: list): 
    #generic_genres = ["blues", "trap", "electro", "house", "dance", "swing" , "orchestra", "cumbia" , "dub", "gospel", "bebop", "bolero", "hardcore", "chanson" ,"adult standards" , "broadway" , "chill" , "classical" , "comedy" , "country", "disco" , "edm" , "emo" , "folk" , "funk" , "hip hop" , "indie" , "jazz" , "latin" , "metal" , "pop" , "punk" , "r&b" , "rap" , "reggae" , "rock" , "soul" , "alternative", "soundtrack" , "worship"]
    #generic_genres = ["rock","pop","hip hop","rap","indie","jazz","country","folk","metal","soul","blues","classical","latin","dance","punk","house","trap","funk","r&b","roots","christian","reggae","tropical","worship","emo","edm","swing","electro","adult standards","disco","electropop","americana","garage"]


    generic_genres_dict = {}

    """
    da se prebroje rijeci
    for genre in genres:
        for word in genre.split():
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    """
    for generic_genre in generic_genres:
        generic_genres_dict[generic_genre] = 0
        for genre in genres:
            if generic_genre in genre:
                generic_genres_dict[generic_genre] += 1

    most_relevant_genre = max(generic_genres_dict, key=generic_genres_dict.get)

    if generic_genres_dict[most_relevant_genre] != 0:
        return most_relevant_genre
    else:
        return ""
    
    
    '''
    for x in generic_genres:
        if generic_genres_dict[x] != 0:
            print (x, generic_genres_dict[x])
    
    print("MAX___")
    max_key = max(generic_genres_dict, key=generic_genres_dict.get)
    print(max_key)
    print("="*11)

    for genre in genres:
        occ = 0
        for generic_genre in generic_genres:
            if generic_genre in genre:
                occ = occ + 1
        print(genre, occ)
    print("========\n")

    '''


def read_genres_(data):
    end = len(data.index)
    sum_others = 0
    i = 1
    others = []
    others_temp = []

    for index, row in data.iterrows():
        """
        genre_list = eval(row["genres"])
        if len(genre_list) > 0:
            assigned_genre, others_temp = assign_super_genre(genre_list)
            if assigned_genre == "other":
                others.append(others_temp)
                sum_others += 1
        """


        assigned_genre = data.iloc[index].genres
        #METAL
        if assigned_genre == "metal":
            song = data.iloc[index]
            if ((song["acousticness"] < 0.1) | 
            (song["danceability"] < 0.5) | 
            (song["energy"] > 0.75) | 
            (song["liveness"] < 0.4) | 
            (song["loudness"] > -10) | 
            (song["speechiness"] < 0.15) | 
            (song["valence"] < 0.6)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"


        #PUNK
        elif assigned_genre == "punk":
            song = data.iloc[index]
            if ((song["acousticness"] < 0.3) | 
            (song["danceability"] < 0.6) | 
            (song["energy"] > 0.6) | 
            (song["instrumentalness"] < 0.5) | 
            (song["liveness"] < 0.4) | 
            (song["loudness"] > -14) | 
            (song["speechiness"] < 0.2)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"


        #BLUES
        elif assigned_genre == "blues":
            song = data.iloc[index]
            if ((song["instrumentalness"] < 0.05) | 
            (song["liveness"] < 0.3) | 
            (song["loudness"] > -14) | 
            (song["speechiness"] < 0.2) | 
            (song["valence"] > 0.3)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"

        #JAZZ
        elif assigned_genre == "jazz":
            song = data.iloc[index]
            if ((song["acousticness"] > 0.4) | 
            (song["danceability"] > 0.4) | 
            (song["energy"] > 0.2) | 
            (song["instrumentalness"] < 0.8) | 
            (song["liveness"] < 0.3) | 
            (song["loudness"] > -20) | 
            (song["speechiness"] < 0.1) |
            (song["popularity"] > 20)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"


        #CLASSICAL
        elif assigned_genre == "classical":
            song = data.iloc[index]
            if ((song["acousticness"] > 0.7) | 
            (song["danceability"] < 0.5) | 
            (song["energy"] < 0.5) | 
            (song["instrumentalness"] > 0.4) | 
            (song["liveness"] < 0.3) | 
            (song["loudness"] < -10) | 
            (song["speechiness"] < 0.1) | 
            (song["valence"] < 0.5) | 
            (song["popularity"] < 0.6)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"


        #LATINO
        elif assigned_genre == "latino":
            song = data.iloc[index]
            if ((song["acousticness"] < 0.6) | 
            (song["danceability"] > 0.5) | 
            (song["energy"] > 0.5) | 
            (song["instrumentalness"] < 0.1) | 
            (song["liveness"] < 0.3) | 
            (song["valence"] > 0.6) | 
            (song["popularity"] > 30)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"
        
        #COUNTRY
        elif assigned_genre == "country":
            song = data.iloc[index]
            if ((song["acousticness"] > 0.2) | 
            (song["danceability"] > 0.5) | 
            (song["energy"] > 0.4) | 
            (song["instrumentalness"] < 0.02) | 
            (song["liveness"] < 0.3) | 
            (song["speechiness"] < 0.1) |
            (song["valence"] < 0.7)): 
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"
        
        #HIP-HOP/RAP
        elif assigned_genre == "hip-hop/rap":
            song = data.iloc[index]
            if ((song["acousticness"] < 0.3) | 
            (song["danceability"] > 0.6) | 
            (song["energy"] > 0.5) | 
            (song["instrumentalness"] < 0.1) | 
            (song["liveness"] < 0.4) | 
            (song["speechiness"] < 0.4) | 
            (song["valence"] > 0.4) | 
            (song["popularity"] > 40)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"
        
        #FOLK
        elif assigned_genre == "folk":
            song = data.iloc[index]
            if ((song["acousticness"] > 0.6) | 
            (song["danceability"] > 0.4) | 
            (song["energy"] > 0.15) | 
            (song["instrumentalness"] < 0.3) | 
            (song["liveness"] < 0.3) | 
            (song["loudness"] > -20) | 
            (song["speechiness"] < 0.1) | 
            (song["valence"] > 0.5)):
                data['genres'].iloc[index] = assigned_genre
            else:
                data['genres'].iloc[index] = "NO_PARAMETER_FIT"

        #TALK
        else:
            if data['speechiness'].iloc[index] > 0.5:
                data['genres'].iloc[index] = "talk"
            else:
                data['genres'].iloc[index] = assigned_genre
        #else:
        #    data["genres"].iloc[index] = "NO_ASSIGNED_GENRE"
        
        sys.stdout.write("\033[F") #back to previous line 
        sys.stdout.write("\033[K") #clear line 
        print((i/end)*100)
        i += 1

    #indexEmpty = data[(data["genres"] == "NO_ASSIGNED_GENRE") | (data["genres"] == "NO_PARAMETER_FIT")].index
    #data.drop(indexEmpty, inplace=True)
    

    #data = deal_with_others(data)

    #print("-"*30)
    #print("NUM OF OTHERS: ")
    #print(sum_others)
    #print("-"*30)
    #print(data)
    #print(others)

    data.to_pickle('data.pkl')

def assign_super_genre(genres: list):
     
    super_genres_dict = {
        "metal": ["metal", "grindcore", "metalcore", "nu-metalcore", "thrash", "power metal"],
        "punk": ["punk", "anarcho-punk", "cowpunk", "oi"],
        "blues": ["blues", "boogie-woogie", "boogie"],
        "jazz": ["jazz", "swing", "bop", "ragtime", "dixieland", "schlager"],
        "classical": ["orchestra", "orchestral", "classical", "baroque", "chamber", "renaissance", "piano", "romantic", "romanticism", "symphony", "violin"],
        "latino": ["latino", "mariachi", "reggaeton", "cumbia", "salsa", "cha-cha-cha", "tango", "bachata", "mexican", "samba", "merenge", "flamenco", "rumba", "cuban", "latin"],
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

#def deal_with_others(data: pd.DataFrame):


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


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        
        if axes is None:
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(title)
        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                        train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                            fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        return plt

def dana_genre(data):  
    x = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "duration_ms"]]
    genres = []
    for index,row in x.iterrows():
        if row['speechiness'] >= 0.22:
            genres.append("rap")
        else:
            if row['danceability'] <= 0.65:
                genres.append("rock")
            else:
                if row['tempo'] >= 115:
                    genres.append("edm")
                else:
                    if row["duration_ms"] >= 280000:
                        genres.append("r&b")
                    else:
                        if row['danceability'] >= 0.70:
                            genres.append("latin")
                        else:
                            genres.append("pop")
                            
    y = pd.DataFrame(genres,columns =['genres'])
    return x, y

def do_PCA(n_comp, x_train, x_test):
    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit(x_train)
    train = pca.transform(x_train)
    test = pca.transform(x_test)
    return train, test

def read_ext_data(data_path):
    data_df = pd.read_csv(data_path)
    x = data_df[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "duration_ms"]]
    names = data_df[["artists", "name", "year"]]
    return x, names

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


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    #read_genres()
    #data__ = pd.read_csv("data_w_genres.csv")
    #read_genres_(data__)
    #data_w_genres = pd.read_pickle('data.pkl')
    #whole_data_wo_genres = pd.read_csv('data.csv')
    #data = add_generic_genres_to_data(data_w_genres, whole_data_wo_genres) 
    dataa = pd.read_csv("data_w_added_genres_1.csv")
    read_genres_(dataa)
    data = pd.read_pickle("data.pkl")
    print("* Done readin' feller!")
    
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
    


  
    data = data.drop(columns=['artists'])
    x_train = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "liveness", "popularity"]]


    
    y_train = data['genres']
    #x_train, y_train = dana_genre(data)
    x_train = normalize(x_train)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)
    print("* Done splittin'")

    
    """ 
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
    classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', max_depth=100)
    #classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=4)
    #classifier = KNeighborsClassifier(n_neighbors=1)
    
    
   

    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    """
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
    
