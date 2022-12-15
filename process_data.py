import pandas as pd
import player_scrape as ps
from sklearn.model_selection import train_test_split

def get_stats(df):
    # print(df)
    column_names = list(df.columns.values)
    # print(column_names)
    
def get_player_names(df):
    global name_set
    name_set = set()
    for item in df['player_name'].iteritems():
        name_set.add(item[1])

def shot_clock_map(time):
    if pd.isna(float(time)):
        return 0
    else:
        return time

def map_data(df):
    get_player_names(df)
    win_map = {'W':1, 'L':0}
    loc_map = {'H':1, 'A':0}
    shot_made_map = {'made':1, 'missed': 0}
    two_pt_map = {2:1, 3: 0}
    name_dict = ps.get_ratings(name_set)
    new_wins = df['W'].map(win_map)
    new_loc = df['LOCATION'].map(loc_map)
    new_shot_made = df['SHOT_RESULT'].map(shot_made_map)
    new_2_pt = df['PTS_TYPE'].map(two_pt_map)
    new_player_name = df['player_name'].map(name_dict)
    df.update(new_wins)
    df.update(new_loc)
    df.update(new_shot_made)
    df.update(new_2_pt)
    df.update(new_player_name)
    df['SHOT_CLOCK'] = df['SHOT_CLOCK'].map(lambda time : shot_clock_map(time))
    df['GAME_CLOCK'] = df['GAME_CLOCK'].map(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))

def convert_to_logisticsvm(df):
    global X_train
    global y_train
    global X_test
    global y_test
    global X_names
    map_data(df)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_names = df.columns.to_list()
    X_names.remove('SHOT_RESULT')
    y_name = 'SHOT_RESULT'
    X_train = train[X_names].values
    y_train = train[y_name].values
    y_train = y_train.astype('int')
    X_test = test[X_names].values
    y_test = test[y_name].values
    y_test = y_test.astype('int')

def go():
    convert_to_logisticsvm(processed_df)


nba_df = pd.read_csv("./shot_logs.csv")
processed_df = nba_df.drop(['MATCHUP', 'GAME_ID', 'player_id', 'CLOSEST_DEFENDER_PLAYER_ID', 'FGM', 'PTS', 'CLOSEST_DEFENDER'], axis = 1)

# get_stats(processed_df)

# go()

# print(name_dict)
