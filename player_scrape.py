from types import NoneType
import requests
from bs4 import BeautifulSoup

url = "https://hoopshype.com/nba2k/2014-2015/"

response = requests.get(url)
html = response.text

soup = BeautifulSoup(html, "html.parser")

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



def get_ratings(name_set):
    make_name_dict()
    new_dict = {}
    for name in name_set:
        if name in name_dict:
            new_dict[name] = name_dict[name]

    return new_dict