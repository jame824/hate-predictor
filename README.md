# hate-predictor

explanations of files in the repository:

embedding_df.py and embedding_pt.py - processes, labels, and embeds the text of Reddit users from .json files. embedding_df stores the data in a pandas data frame while embedding_pt directly puts the data into a pytorch dataset. 

embedding_server.py - version edited to run on the donut cluster. edited to have data embedded and added to data frame as a tensor one at a time to preserve memory and only use 30000 users each from specified directories. needs further editing such as editing to not include users with under 2 posts.

firstuserid.py - used on a dataset containing r/WhiteRights comments filtered 60 days before and after the Charlottesville Riot, creates a matplotlib graph of new user counts through the time period.

grab_all_authors.py - obtains the names of all authors in a specified directory and outputs them into a comma-delimited .txt file.

grab_50k.py - obtains 50k random users from the .txt file created by grab_all_authors.py and outputs to a .txt file

grab_json.py - uses grab_50k's .txt file to create .json files for users from .zst files containing reddit comments and submissions

hate_user_grab.py - script assumes there are folders in the specified directories named in the format "(subreddit)_Users_(Submissions/Comments)," looks to grab 50k users' comments first then grabs those same users' submissions from the other directory if it exists.

model.py and model_server.py - testing versions of model, uses the .pt dataset as an input from the embedding files

model_server_edit.py - model ran on the server, needs more clarification on validation, training, and testing sets.

test.py - displays output of .zst file, script made just to further understand how to work with .zst files
