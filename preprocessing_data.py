import random
import numpy as np
import pandas as pd
import pickle
import scipy
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def load_problem(file_name = "data.pickle"):
    f_myfile = open(file_name, 'rb')
    data = pickle.load(f_myfile)
    f_myfile.close()
    return data["x_train"], data["y_train"],data["x_test"], data["y_test"]

def read_and_combine_data(files, features):
    frames = []
    for f in files:
        file = pd.read_csv(f)[features]
        frames.append(file)
    return pd.concat(frames).fillna("-")

def save_as_picke(data,filename = "data.pickle"):
    with open(filename, 'wb') as outfile:
        pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
def age_conversion(p):
    p["dob"] = pd.to_datetime(p["dob"])
    temp = pd.DataFrame({'year': [1980],
                        'month': [1],
                        'day': [1]})
    temp = pd.to_datetime(temp)

    p["dob"] = p["dob"].fillna(temp)
    p["age"] = pd.Timestamp('today')
    p["age"] = (p["age"] - p["dob"])/ np.timedelta64(1, 'Y')
    return p
def union_batter_pitcher(p,f):
    #last first throws bats height weight dob
    p = p.copy()
    p.rename(columns={'bref_id': 'pitcher', 'last': 'p_last', 'first':'p_first', 'height': 'p_height', 'weight':'p_weight', 'age':'p_age'}, inplace=True)
    # result = pd.concat([f, p], axis=1, join='inner')

    combined = pd.merge(f, p[["pitcher", "p_last", "p_first", "p_height", "p_weight", "p_age", "throws"]], on='pitcher')
    p.rename(columns={'pitcher': 'batter', "p_last":"b_last", "p_first":"b_first", "p_height":"b_height", "p_weight":"b_weight", "p_age":"b_age"}, inplace=True)
    # result = pd.merge(result, p, on='batter')
    combined = pd.merge(combined, p[["batter", "b_last", "b_first", "b_height", "b_weight", "b_age", "bats"]], on='batter')
    combined.head()
    return combined
def labe_encoder_conversion(data,features):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    return np.array([LabelEncoder().fit_transform(data[:,i]) for i in range(data.shape[1])])

def one_hot_encoding_conversion(data,features_for_LE_and_OH, extra_features_for_OH,features_rest, train_sz,label = "umpcall",NormalizeRest = True):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    OH_enc = OneHotEncoder(sparse = True)
    LE_result = labe_encoder_conversion(data[features_for_LE_and_OH].fillna("-").as_matrix(),features_for_LE_and_OH)
    LE_and_extra = np.append(LE_result.transpose(),data[extra_features_for_OH].fillna("-"),axis=1)
    OH_result = OH_enc.fit_transform(LE_and_extra)
    rest = data[features_rest]
    if NormalizeRest:
        rest = data[features_rest]
        rest_train = rest[:train_sz]
        rest_test = rest[train_sz:]
        scaler = MinMaxScaler()
        rest_train = scaler.fit_transform(rest_train)
        rest_test = scaler.transform(rest_test)
        rest = np.append(rest_train,rest_test,axis = 0)
    x = scipy.sparse.hstack((OH_result,rest))
    y = data[label].as_matrix()
    return x.tocsr(),y
def generate_data(train_years, test_years, fx_features_to_keep,\
                                features_for_LE_and_OH,\
                                extra_features_for_OH,\
                                features_rest,\
                                base_dir = "Data/",\
                                player_filename = "MLB_Players.csv",\
                                label = "umpcall",\
                                filename = "data.pickle",
                                post_season = True,
                                toShuffle = True,
                                NormalizeRest = True):

    train_regular_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in train_years]
    train_post_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in train_years]

    test_regular_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in test_years]
    test_post_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in test_years]

    if post_season:
        train_data = read_and_combine_data(train_regular_season+train_post_season,fx_features_to_keep)
        test_data = read_and_combine_data(test_post_season+test_post_season,fx_features_to_keep)
    else:
        train_data = read_and_combine_data(train_regular_season,fx_features_to_keep)
        test_data = read_and_combine_data(test_regular_season,fx_features_to_keep)
    player = age_conversion(pd.read_csv(base_dir+player_filename))

    train_data = union_batter_pitcher(player,train_data)
    test_data = union_batter_pitcher(player,test_data)

    train_sz = train_data.shape[0]
    train_test = pd.concat([train_data,test_data])
    print("Start One-hot encoding")
    x,y = one_hot_encoding_conversion(train_test,
                                        features_for_LE_and_OH,
                                        extra_features_for_OH,
                                        features_rest,
                                        train_sz,
                                        NormalizeRest = NormalizeRest)
    # shuffle
    if toShuffle:
        x_train, y_train = shuffle(x[:train_sz], y[:train_sz])
        x_test, y_test = shuffle(x[train_sz:], y[train_sz:])
    else:
        x_train, y_train = x[:train_sz], y[:train_sz]
        x_test, y_test = x[train_sz:], y[train_sz:]
    print("Start writing data to {}".format(base_dir+filename))
    save_as_picke({"x_train":x_train,"y_train":y_train,
                    "x_test":x_test,"y_test":y_test},
                    base_dir+filename)
    
def main():
    fx_features_to_keep = ["pitcher","batter", "balls","strikes","pitch_count","inning","side", "umpcall"]
    train_year = [2,3,4]
    test_year = [5]
    player_filename = "MLB_Players.csv"
    features_for_LE_and_OH = ["pitcher","batter","side","throws","bats","throws","bats"]
    extra_features_for_OH = ["inning"]
    features_rest = ["pitch_count","balls","strikes","p_height", "p_weight", "p_age","b_height", "b_weight", "b_age",]
    filename = "{0}|{1}.pickle".format(str(train_year),str(test_year))

    generate_data(train_year,test_year,fx_features_to_keep,
            features_for_LE_and_OH,
            extra_features_for_OH,
            features_rest,filename=filename)
    
    # to read data from pickle
    x_train, y_train, x_test, y_test =  load_problem("Data/"+filename)
if __name__ == '__main__':
  main()

