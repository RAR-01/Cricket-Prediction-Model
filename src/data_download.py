import requests
import zipfile
import io
import os

# IPL dataset URL from Cricsheet
URL = "https://cricsheet.org/downloads/ipl.zip"

# Folder where raw data will be stored
DATA_FOLDER = "data/raw"


def download_ipl_data():
    
    print("Downloading IPL dataset...")

    response = requests.get(URL)

    if response.status_code != 200:
        print("Download failed.")
        return

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Create folder if not exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    zip_file.extractall(DATA_FOLDER)

    print("Dataset downloaded and extracted to data/raw")


if __name__ == "__main__":
    download_ipl_data()