import pandas as pd

import requests
from bs4 import BeautifulSoup

def get_stats(df):
    print(df)
    column_names = list(df.columns.values)
    print(column_names)
    name_set = set()
    for item in df['player_name'].iteritems():
        name_set.add(item[1])

    print(name_set)
    print(len(name_set))

nba_df = pd.read_csv("./shot_logs.csv")
# processed_df = nba_df.drop(['MATCHUP', 'GAME_ID', 'player_id', 'CLOSEST_DEFENDER_PLAYER_ID'], axis = 1)
processed_df = nba_df.drop(['MATCHUP', 'GAME_ID', 'player_id', 'CLOSEST_DEFENDER_PLAYER_ID', 'FINAL_MARGIN', 'PERIOD', 'SHOT_NUMBER', 'CLOSEST_DEFENDER'], axis = 1)

get_stats(processed_df)