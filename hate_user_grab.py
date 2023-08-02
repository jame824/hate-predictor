#find 50k users from directories list ...C:\Users\fujin\Downloads\test\test_json\hate_comments\(list)
#list = ["GenderCritical_Users_Comments", "dolan_Users_Comments", "WhiteRights_Users_Comments", "Delraymisfits_Users_Comments","TheRedPill_Users_Comments","asktrp_Users_Comments","trolling_Users_Comments","ImGoingToHellForThis_Users_Comments","MGTOW_Users_Comments","FuckYou_Users_Comments", "CringeAnarchy_Users_Comments","TrollGC_Users_Comments","CCJ2_Users_Comments","TruFemcels_Users_Comments","Incels_Users_Comments","sjwhate_Users_Comments","fatpeoplehate_Users_Comments","GreatApes_Users_Comments","frenWorld_Users_Comments","opieandanthony_Users_Comments","Neofag_Users_Comments","milliondollarextreme_Users_Comments","CoonTown_Users_Comments","Braincels_Users_Comments","honkler_Users_Comments"]
#match those users and their comments and submissions
#comments exist in the same directories as above, submissions exist in the same names but comments is submissions instead and is in C:\Users\fujin\Downloads\test\test_json\hate_submissions\(list)

import os
import json
import shutil
import random
from tqdm import tqdm

# Define the list of directories
directories = ["asktrp", "Braincels", "CCJ2", "CoonTown", "Delraymisfits", "dolan", "fatpeoplehate", "FuckYou", "frenWorld", "GenderCritical", "GreatApes", "honkler", "Incels", "Neofag", "sjwhate", "TheRedPill", "TrollGC", "TruFemcels", "WhiteRights"]

# Define the base paths for comments and submissions
base_path_comments = "C:\\Users\\fujin\\Downloads\\test\\test_json\\hate_comments\\"
base_path_submissions = "C:\\Users\\fujin\\Downloads\\test\\test_json\\hate_submissions\\"

# Define the destination directory
destination_directory = "C:\\Users\\fujin\\Downloads\\test\\test_json\\hate\\"

# We'll store the users and their data in this dictionary
user_data = {}

# We'll store all the file paths for comments here
comment_file_paths = []

# Loop through all the directories
for directory in directories:
    # Construct the directory names for comments and submissions
    comment_dir = directory + "_Users_Comments"

    # Get the list of files in the directory
    comment_files = os.listdir(base_path_comments + comment_dir)

    # Loop through the files
    for file in comment_files:
        # Add the file path to our list
        comment_file_paths.append(base_path_comments + comment_dir + "\\" + file)

# Now we randomly select 50,000 comment file paths
selected_comment_files = random.sample(comment_file_paths, 50000)

# And process them
for comment_file_path in tqdm(selected_comment_files, desc="Processing files"):
    # Get the username from the filename
    username = os.path.basename(comment_file_path).split('_comments')[0]

    # If the destination file already exists, remove it
    if os.path.exists(destination_directory + "\\" + os.path.basename(comment_file_path)):
        os.remove(destination_directory + "\\" + os.path.basename(comment_file_path))

    # Move the comment file to the destination directory
    shutil.move(comment_file_path, destination_directory)

    # Now find and process the corresponding submission file for this user
    for directory in directories:
        submission_dir = directory + "_Users_Submissions"
        submission_file_path = base_path_submissions + submission_dir + "\\" + username + "_submissions.json"

        if os.path.exists(submission_file_path):
            # If the destination file already exists, remove it
            if os.path.exists(destination_directory + "\\" + os.path.basename(submission_file_path)):
                os.remove(destination_directory + "\\" + os.path.basename(submission_file_path))

            # Move the submission file to the destination directory
            shutil.move(submission_file_path, destination_directory)
            break  # We found the submission file for this user, no need to check the other directories