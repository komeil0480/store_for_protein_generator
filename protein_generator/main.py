print("start of main.py", flush=True)
import os
import sys
import numpy as np
from predict import predict
from git_handler import push_on_git
# import tqdm.notebook
input_directory = sys.argv[1]
my_space_path = sys.argv[2]
git_repo_path = sys.argv[3]
print("before while in main.py")
while True:
    check_again = False
    for folder_name in os.listdir(input_directory):
        # if(folder_name == "1d5x", folder_name == "2m49"):
        #     continue
        folder_path = os.path.join(input_directory, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Define the prediction folder path to check
            prediction_folder_path = os.path.join(folder_path, 'prediction')

            # Check if the prediction folder exists
            if os.path.isdir(prediction_folder_path):
                print(f"Prediction folder exists. Skipping folder {folder_name}.")
            else:
                print(f"Prediction folder does not exist. Processing folder {folder_name}.")
                # Find the .npz file in the folder
                npz_files = [file for file in os.listdir(folder_path) if file.endswith('.npz')]
                #print(npz_files)
                npz_file_path = os.path.join(folder_path, npz_files[0])  # Load the first .npz file
                #print(f"Loading .npz file: {npz_files[0]}")
                np_example = np.load(npz_file_path, allow_pickle=True)
                # Convert np_example to only its key-value pairs
                np_example = {key: np_example[key] for key in np_example.keys()}
                ### Call the prediction function
                predict(folder_path, git_repo_path, np_example)
                ###
                check_again = True
                # push on git
                push_on_git(folder_name, my_space_path)

                break
    if not check_again:
        break


