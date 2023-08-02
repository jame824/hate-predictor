import os
import shutil
import random

n = 10  # number of users

input_dirs = {
    'hate': r"C:\Users\fujin\Downloads\test\test_json\hate",
    'control': r"C:\Users\fujin\Downloads\test\test_json\control"
}

output_dirs = {
    'hate': r"C:\Users\fujin\Downloads\test\test_json\hate_test",
    'control': r"C:\Users\fujin\Downloads\test\test_json\control_test"
}

for category in ['hate', 'control']:
    input_dir = input_dirs[category]
    output_dir = output_dirs[category]

    # Find unique users in this directory.
    users = set()
    for filename in os.listdir(input_dir):
        if "_submissions" in filename or "_comments" in filename:
            user = filename.split('_')[
                0]  # The format is now assumed to be {username}_submissions.json or {username}_comments.json
            users.add(user)

    print(f"Found {len(users)} unique users in {category}.")

    # If there are less users than n, adjust n.
    if len(users) < n:
        n = len(users)

    # Select n random users.
    selected_users = random.sample(list(users), n)
    print(f"Selected users: {selected_users}")

    # Copy all files for the selected users.
    for user in selected_users:
        for suffix in ['submissions', 'comments']:
            filename = f"{user}_{suffix}.json"
            if filename in os.listdir(input_dir):
                shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, filename))
