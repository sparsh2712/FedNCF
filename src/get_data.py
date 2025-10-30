import zipfile
import urllib.request

# download and extract
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
urllib.request.urlretrieve(url, "ml-1m.zip")

with zipfile.ZipFile("ml-1m.zip", "r") as zip_ref:
    zip_ref.extractall(".")