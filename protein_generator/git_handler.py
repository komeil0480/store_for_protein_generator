import subprocess
import os
def push_on_git(protein_name, my_space_path):

  # Change to the target directory
    os.chdir(my_space_path)
    
    # Add all files to the staging area
    subprocess.run(["git", "add", "."], check=True)
    
    # Commit the changes
    commit_text = "prediction folder for " + protein_name + " append."
    subprocess.run(["git", "commit", "-m", commit_text], check=True)
    
    # Push the changes to the remote repository
    subprocess.run(["git", "push"], check=True)
    
    # Optionally, change back to the previous directory if needed.
    # os.chdir("..")
