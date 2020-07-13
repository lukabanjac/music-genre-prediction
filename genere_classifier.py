import pandas as pd
import numpy as np
import ast

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

import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def read_data(data: pd.DataFrame):
    x = data[
        ["danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
         "valence", "tempo", "liveness"]]
    y = data['genres'].str.lower()
    #generic_genres = [ "blues", "r&b", "classical", "country", "dance", "house", "hip hop", "indie",
    # "jazz", "pop", "dub", "adult standards", "chill", "rock", "punk", "metal", "broadway", "latin", "reggae", "rap", "disco"]

    generic_genres = ["blues", "banda", "electro", "house", "dance", "swing" , "orchestra", "cumbia" , "dub", "gospel", "bebop", "bolero", "hardcore", "chanson" ,"adult standards" , "broadway" , "chill" , "classical" , "comedy" , "country", "disco" , "edm" , "emo" , "folk" , "funk" , "hip hop" , "indie" , "jazz" , "latin" , "metal" , "pop" , "punk" , "r&b" , "rap" , "reggae" , "rock" , "soul" , "psy", "soundtrack" , "worship"]

    
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
    pca = PCA(n_components=n_comp)
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

if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    data = pd.read_pickle('data.pkl')
    #print(data)
    print(len(data['genres'].unique().tolist()))
    #x_train, y_train = read_data(data)
    x_train, y_train = dana_genre(data)

    #x_train = normalize(x_train)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)

    """ 
    #HEATMAP
    fig = plt.figure(figsize=(13,6))
    labels = x_train.columns.tolist()
    ax1 = fig.add_subplot(111)
    ax1.set_xticklabels(labels,rotation=90, fontsize=10)
    ax1.set_yticklabels(labels,fontsize=10)
    plt.imshow(x_train.corr(), cmap='hot', interpolation='nearest')
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
    """


    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)


    


    print(x_train)
    print(x_test)
    
    title = "Learning curve"
    
    #PCA
    #x_train, x_test = do_PCA(8, x_train, x_test)

    #classifier =  SVC(kernel='linear',gamma=0.01, C=0.01)
    #classifier = GradientBoostingClassifier(max_depth=3, n_estimators=700)
    classifier = RandomForestClassifier(n_estimators=200, criterion='gini', max_features='auto', max_depth=100)
    #aclassifier = KNeighborsClassifier(n_neighbors=1)
    
    #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    #fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    #plot_learning_curve(classifier, title, x_train, y_train, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    #plt.show()

    #classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=4)
    classifier.fit(x_train, y_train)
    print(classifier.score(x_train,y_train))
    y_pred = classifier.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)

    x_ext, other_ext = read_ext_data("data.csv")

    x_ext = sc.transform(x_ext)

    y_pred_ext = classifier.predict(x_ext)

    other_ext.insert(loc=3, column="predicted_genres", value=y_pred_ext)

    other_ext.to_csv("predicted_data.csv")
    print(other_ext.nunique())


    
