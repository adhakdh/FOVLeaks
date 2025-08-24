import gdown
import zipfile
import json
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.append(parent_dir)

if __name__ == "__main__":
    with open(f'../lib/datastet.json', 'r') as f:
        dataset = json.load(f)

Dataset_dict = dataset["Dataset"]

current_file_path = os.path.abspath(__file__)
current_folder_path = os.path.dirname(current_file_path)
current_folder_name = os.path.basename(current_folder_path)

if current_folder_name in Dataset_dict:
    file_id = Dataset_dict[current_folder_name]  
    destination = f'Dataset_NeckPass.zip'  
    gdown_link = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(gdown_link, destination, quiet=False)

    ExtractPath = os.getcwd()
    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(ExtractPath)