from types import NoneType
import requests
from bs4 import BeautifulSoup

url = "https://hoopshype.com/nba2k/2014-2015/"

response = requests.get(url)
html = response.text

soup = BeautifulSoup(html, "html.parser")

# use bs4 to scrape player ratings off of the internet
def make_name_dict():
    # Find the table containing the player ratings
    table = soup.find("table", {"class": "hh-salaries-ranking-table"})

    tbody = table.find("tbody")

    # Extract the rows of the table
    rows = tbody.find_all("tr")

    # Iterate over the rows and extract the player name and rating
    global name_dict
    name_dict = {}
    for row in rows:
        # Find the name and rating cells in the row
        name_cell = row.find("td", {"class": "name"})
        rating_cell = row.find("td", {"class": "value"})

        # Extract the player name and rating from the cells
        name_dict[str(name_cell.text.strip()).lower()] = int(rating_cell.text.strip())

        # manually fixing typos in dataset
        name_dict["greg smith"] = 61
        name_dict["dirk nowtizski"] = 84
        name_dict["time hardaway jr"] = 69
        name_dict["nikola mirotic"] = 75
        name_dict["mo williams"] = 74
        name_dict["steve adams"] = 74
        name_dict["mnta ellis"] = 82
        name_dict["travis wear"] = 66
        name_dict["dwayne wade"] = 86
        name_dict["al farouq aminu"] = 74
        name_dict["dennis schroder"] = 70
        name_dict["kostas papanikolaou"] = 73
        name_dict["nene hilario"] = 79
        name_dict["shawne williams"] = 66
        name_dict["carlos boozer"] = 78
        name_dict["nerles noel"] = 76
        name_dict["lou williams"] = 77
        name_dict["beno urdih"] = 75
        name_dict["jon ingles"] = 69
        name_dict["hedo turkoglu"] = 71
        name_dict["kyle oquinn"] = 71
        name_dict["jimmer dredette"] = 72
        name_dict["danilo gallinai"] = 77
        name_dict["alan crabbe"] = 67
        name_dict["joey dorsey"] = 70
        name_dict["jerome jordan"] = 70
        name_dict["rasual butler"] = 71
        name_dict["jason maxiell"] = 68

# method to set the ratings
def get_ratings(name_set):
    make_name_dict()
    new_dict = {}
    for name in name_set:
        if name in name_dict:
            new_dict[name] = name_dict[name]
    return new_dict