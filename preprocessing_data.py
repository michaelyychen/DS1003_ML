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
def gen_pitch_type_feature_name(pitch_types):
    return ["p_"+name for name in pitch_types]+["p_"+name+"_speed" for name in pitch_types]
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
def get_pitch_distribution(player_data, pitch_types):
    pitch_stats = {}
    for index , row in player_data.iterrows():
        pitch_stats[row.bref_id] = {p_type:0 for p_type in pitch_types}
        for p_type in pitch_types:
            pitch_stats[row.bref_id][p_type] = row[p_type]
    return pitch_stats
def test_data_pitch_type_conversion(test_data, player_data,pitch_types):
    import random
    name2pitch_distrib = get_pitch_distribution(player_data,pitch_types)
    def get_pitch(name):
        player = name2pitch_distrib[name]
        if "weight" not in player:
            player["weight"] = [player[p_type] for p_type in pitch_types]
        w = player["weight"]
        if np.sum(w)>0 :
            decision = random.choices(pitch_types,weights = player["weight"])
            decision = decision[0]
        else :
            decision = "FF"
        return decision

    def name_to_pitch(row):
        p = row["pitcher"]
        return get_pitch(p)
    test_data["pitch_type"] = test_data.apply(name_to_pitch,axis = 1)
    return test_data

def union_batter_pitcher(p,f):
    #last first throws bats height weight dob
    p = p.copy()
    # print(p.head())
    p_columns = []
    b_columns = []
    for column in p.columns:
        p_columns.append("p_" +  column)
    p_columns[2] = "pitcher"
    p.columns = p_columns
    # p.rename(columns={'bref_id': 'pitcher', 'last': 'p_last', 'first':'p_first', 'height': 'p_height', 'weight':'p_weight', 'age':'p_age', 'hit_ratio':'p_hit_ratio'}, inplace=True)
    # result = pd.concat([f, p], axis=1, join='inner')

    combined = pd.merge(f, p[["pitcher", "p_throws","p_hit_ratio", "p_strike_ratio"]], on='pitcher')

    for column in p.columns:
        b_columns.append("b_" +  column[2:])
    b_columns[2] = "batter"
    p.columns = b_columns
    # print(b_columns)
    # p.rename(columns={'pitcher': 'batter', "p_last":"b_last", "p_first":"b_first", "p_height":"b_height", "p_weight":"b_weight", "p_age":"b_age", "p_hit_ratio" : "b_hit_ratio"}, inplace=True)
    # result = pd.merge(result, p, on='batter')
    combined = pd.merge(combined, p[["batter", "b_bats", "b_hit_ratio"]], on='batter')

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
    if NormalizeRest and len(features_rest) >0 :
        rest = data[features_rest]
        rest = data[features_rest]
        rest_train = rest[:train_sz]
        rest_test = rest[train_sz:]
        scaler = MinMaxScaler()
        rest_train = scaler.fit_transform(rest_train)
        rest_test = scaler.transform(rest_test)
        rest = np.append(rest_train,rest_test,axis = 0)
        x = scipy.sparse.hstack((OH_result,rest))
    else:
        x = OH_result
    y = data[label].as_matrix()
    return x.tocsr(),y
def generate_data(train_years, test_years, fx_features_to_keep,
                                features_for_LE_and_OH,
                                extra_features_for_OH,
                                features_rest,
                                pitch_types,
                                base_dir = "Data/",
                                player_filename = "MLB_Players.csv",
                                label = "umpcall",
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

    test_data = test_data_pitch_type_conversion(test_data,player,pitch_types)

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
    pitch_types = ["FF","SL","SI","CH","FT","CU","KC","FC","FS"]
    fx_features_to_keep = ["pitcher","batter", "pitch_type", "balls","strikes","pitch_count","inning","side", "umpcall"]
    train_year = [2,3,4]
    test_year = [5]
    player_filename = "MLB_Players_Stats.csv"
    features_for_LE_and_OH = ["pitcher","batter","side","p_throws","b_bats","pitch_type"]
    extra_features_for_OH = ["inning"]
    # features_rest = ["pitch_count","balls","strikes","p_height", "p_weight", "p_age","b_height", "b_weight", "b_age","p_hit_ratio","b_hit_ratio"]+\
    #                 gen_pitch_type_feature_name(pitch_types)
    features_rest = ["pitch_count","balls","strikes","p_hit_ratio","b_hit_ratio"]
    filename = "save.pickle"

    generate_data(train_year,test_year,fx_features_to_keep,
            features_for_LE_and_OH,
            extra_features_for_OH,
            features_rest,
            pitch_types,
            filename=filename,
            player_filename = player_filename)

    # to read data from pickle
    x_train, y_train, x_test, y_test =  load_problem("Data/"+filename)
if __name__ == '__main__':
  main()
