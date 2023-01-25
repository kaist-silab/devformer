import os
import pickle
import zipfile

from src.utils.downloader import download_url

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), "wb") as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), "rb") as f:
        return pickle.load(f)


def download_data(url="https://drive.google.com/uc?id=1cANSJRW7STCl_7cWacDajWMXcEUQG1SK",
                 md5="378339c8bf144e48d57db1121b46ccfe",
):
    """Download data from Google Drive and unzip it in data/ directory."""
    download_url(url, ".", md5=md5, filename="data.zip")   
    print("Extracting data.zip...")
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove("data.zip")
    print("Done!")


def check_data_is_downloaded():
    """Check if data is downloaded, otherwise download it."""
    # just check if data/dpp/test_data.pkl exists
    if not os.path.isfile("data/dpp/test_data.pkl"):
        return False
    return True
