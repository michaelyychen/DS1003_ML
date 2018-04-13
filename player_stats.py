import preprocessing_data as prep
import pandas as pd
import numpy as np

def gen_players_stats(all_data):
    pitch_types = all_data.pitch_type.unique()
    pitchers_name_s = all_data.pitcher.unique()
    batters_name_s = all_data.batter.unique()
    player_stats = {}
    
    def gen_empty_pitch_type(types):
        result = {}
        for t in types:
            result[t] = {
                "count":0,
                "start_speed":0,
                "spin_dir":0,
                "umpcall":{
                    'X':0,
                    'S':0,
                    'B':0
                }
            }
        return result
    def new_player_stat():
        result = {"bat_count":0, 
                "pitch_count":0,
                "hit_count":0, 
                "pitch_type":gen_empty_pitch_type(pitch_types),
                "at_bats":0}
        return result
    old_batter = ""
    init = True
    batter = ""
    #Start iterate through each row in data
    for row in all_data.itertuples(index=False):
        if not init:
            old_batter = batter
        batter = row.batter
        pitcher = row.pitcher
        
        # do batter first
        if batter not in player_stats:
            player_stats[batter] = new_player_stat()
    
        if old_batter != batter:
            player_stats[batter]["at_bats"] += 1
        player_stats[batter]["bat_count"] += 1
        
        if row.umpcall is 'X':
            player_stats[batter]["hit_count"] +=1
        
        
        
        # do pitcher 
        
        if pitcher not in player_stats:
            player_stats[pitcher] = new_player_stat()
        player_stats[pitcher]["pitch_count"] += 1
        
        p = player_stats[pitcher]["pitch_type"][row.pitch_type]
        p["count"]+=1
        p["start_speed"]+=row.start_speed
        p["spin_dir"]+=row.spin_dir
        p["umpcall"][row.umpcall]+=1
        init = False
    
    # Start sumarizing collected data (e.g. calculate average etc.)
    for player,stat in player_stats.items():
        strike_count = 0
        for pitch_type,pitch_type_val in stat["pitch_type"].items():
            if(pitch_type_val["count"]>0):
                pitch_type_val["avg_start_speed"] = pitch_type_val["start_speed"]/pitch_type_val["count"]
                pitch_type_val["avg_spin_dir"] = pitch_type_val["spin_dir"]/pitch_type_val["count"]
                pitch_type_val["possibility"] = pitch_type_val["count"]/stat["pitch_count"]
                strike_count +=pitch_type_val["umpcall"]["S"]
            else:
                pitch_type_val["avg_start_speed"] = 0
                pitch_type_val["avg_spin_dir"] = 0       
                pitch_type_val["possibility"] = 0
        if stat["pitch_count"] > 0:
            stat["strike_ratio"] = strike_count / stat["pitch_count"]
        else:
            stat["strike_ratio"] = 0
            
        if stat["bat_count"] > 0:
            stat["hit_ratio"] = stat["hit_count"] / stat["bat_count"]
        else:
            stat["hit_ratio"] = 0
    return player_stats, pitch_types

def combine_stats_with_original_player_data(players, player_stats,pitch_types):
    
    def new_col_names(pitch_types, rest):
        col = [pth for pth in pitch_types]
        col += [pth+"_speed" for pth in pitch_types]
        col += rest
        return col
    for c  in new_col_names(pitch_types, ["hit_ratio","strike_ratio"]):
        players[c] = 0 
    players["most_likely_pitch"] = "FF"
    temp = players.copy()
    temp = temp.drop(temp.index[0:])

    for index, player in players.iterrows():
        pid = player["bref_id"]
        if pid in player_stats:
            most_like = 0
            plyer = player_stats[pid]
            for pitch_type,pitch_type_val in plyer["pitch_type"].items():
                if(pitch_type_val["count"]>0):
                    player[pitch_type] = pitch_type_val["possibility"]
                    player[pitch_type+"_speed"] = pitch_type_val["avg_start_speed"]
                    if pitch_type_val["possibility"] > most_like:
                        player["most_likely_pitch"] = pitch_type
                        most_like = pitch_type_val["possibility"]
            player["hit_ratio"] = plyer["hit_ratio"]
            player["strike_ratio"] = plyer["strike_ratio"]
        temp = temp.append(player)
    return temp

def main():
    years = [2,3,4]
    base_dir = "Data/"
    output_filename = base_dir+"MLB_Players_Stats.csv"
    features_to_load = ["pitcher","batter","pitch_type","start_speed","spin_dir","umpcall"]
    train_regular_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in years]
    train_post_season = [base_dir+"MLB_201{0}/MLB_PitchFX_201{0}_RegularSeason.csv".format(i) for i in years]
    all_data = prep.read_and_combine_data(train_regular_season+train_post_season,features_to_load)
    players = pd.read_csv(base_dir+"MLB_Players.csv")

    stats,types =  gen_players_stats(all_data)
    new_players = combine_stats_with_original_player_data(players,stats,types)
    print("Stats Collected from ",years)
    print("Types include: ",types)
    new_players.to_csv(output_filename )
    print("CSV saved as \"{}\"".format(output_filename))
if __name__ == '__main__':
  main()