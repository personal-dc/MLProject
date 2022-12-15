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


def update_shot_clock(ds):
    new_col = []
    for index, value in ds['SHOT_CLOCK'].iteritems():
        if pd.isna(value):
            new_col.append(ds['GAME_CLOCK'][index])
        else:
            new_col.append(value)
    ds['SHOT_CLOCK'] = new_col

def map_data(df):
    get_player_names(df)
    name_dict = ps.get_ratings(name_set)
    df['player_name'] = df['player_name'].map(lambda name : name_dict.get(name))
    df['W'] = df['W'].map(lambda win : 1 if win == 'W' else 0)
    df['LOCATION'] = df['LOCATION'].map(lambda loc : 1 if loc == 'H' else 0)
    df['PTS_TYPE'] = df['PTS_TYPE'].map(lambda type : 1 if type == 2 else 0)
    df['SHOT_RESULT'] = df['SHOT_RESULT'].map(lambda res : 1 if res == 'made' else 0)
    df['GAME_CLOCK'] = df['GAME_CLOCK'].map(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
    update_shot_clock(df)

def convert_to_logisticsvm(df):
    global X_train_sklearn
    global y_train_sklearn
    global X_test_sklearn
    global y_test_sklearn
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    X_names = df.columns.to_list()
    X_names.remove('SHOT_RESULT')
    y_name = 'SHOT_RESULT'
    X_train_sklearn = train[X_names].values
    y_train_sklearn = train[y_name].values
    y_train_sklearn = y_train_sklearn.astype('int')
    X_test_sklearn = test[X_names].values
    y_test_sklearn = test[y_name].values
    y_test_sklearn = y_test_sklearn.astype('int')

def convert_to_XGB(df):
    global X_train_xgb
    global y_train_xgb
    global X_test_xgb
    global y_test_xgb
    global X_validation_xgb
    global y_validation_xgb
    X_names = df.columns.to_list()
    X_names.remove('SHOT_RESULT')
    y_name = 'SHOT_RESULT'
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(df[X_names] , df[y_name], test_size=0.50, random_state=42)
    X_validation_xgb, X_test_xgb, y_validation_xgb, y_test_xgb = train_test_split(df[X_names] , df[y_name], test_size=0.50, random_state=42)


def go():
    map_data(processed_df)
    convert_to_logisticsvm(processed_df)
    convert_to_XGB(processed_df)


nba_df = pd.read_csv("./shot_logs.csv")
processed_df = nba_df.drop(['MATCHUP', 'GAME_ID', 'player_id', 'CLOSEST_DEFENDER_PLAYER_ID', 'FGM', 'PTS', 'CLOSEST_DEFENDER'], axis = 1)

# get_stats(processed_df)

# go()

# print(name_dict)
