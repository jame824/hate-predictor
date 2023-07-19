import pandas as pd
import json
import os
from transformers import LongformerTokenizerFast, LongformerModel
import torch
from torch.utils.data import TensorDataset
#tqdm for progress
from tqdm import tqdm

# structure of the program:
# pre-process text,
# combine comments and submissions, if hate, remove all posts after first joining hate subreddit, then concatenate all text for users... pandas .readjson, write a new dataframe, keep in ram (sort by time and find earliest time)
# if from (hate folder)/(non-hate) then this label is added to the df entry for the user
# feed into longformers
# add array produced by longformers with label to df

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")

model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

model.to(DEV)

def process_text(text):
    # General purpose function for cleaning reddit text for transformer model input
    text = str(text)
    text = text.replace(r'&amp;?', r'and')
    text = text.replace(r'&lt;', r'<')
    text = text.replace(r'&gt;', r'>')

    # insert space between punctuation marks
    text = text.replace(r'([\w\d]+)([^\w\d ]+)', r'\1 \2')
    text = text.replace(r'([^\w\d ]+)([\w\d]+)', r'\1 \2')

    text = text.strip()

    return text

def preprocess_text(directory, subreddits_path):
    # Initialize a list to hold all dataframes
    dfs = []

    # Iterate over all JSON files in the directory
    for file_name in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        # Read JSON file
        # Read JSON file
        with open(file_path, 'r') as f:
            # Read the file line by line and use json.loads() on each line
            data = []
            for line in f:
                data.append(json.loads(line))

        # Convert JSON to DataFrame
        df = pd.DataFrame(data)

        # Check if the 'body', 'selftext', and 'title' columns exist before trying to apply process_text
        if 'body' in df.columns:
            df['body'] = df['body'].apply(process_text)
        else:
            df['body'] = ''

        if 'selftext' in df.columns:
            df['selftext'] = df['selftext'].apply(process_text)
        else:
            df['selftext'] = ''

        if 'title' in df.columns:
            df['title'] = df['title'].apply(process_text)
        else:
            df['title'] = ''

        # Combine 'body', 'selftext', and 'title' into a new 'text' column
        df['text'] = df[['body', 'selftext', 'title']].apply(lambda row: ' </s> '.join(row.values.astype(str)), axis=1)

        # Add df to list
        dfs.append(df)

    # Concatenate all dataframes in the list
    df_all = pd.concat(dfs, ignore_index=True)

    # Convert 'created_utc' to datetime
    df_all['created_utc'] = pd.to_datetime(df_all['created_utc'], unit='s')

    # If the directory is 'hate', retain only the posts before the first mention of a subreddit from the text file
    if 'hate' in directory:
        with open(subreddits_path, 'r') as f:
            subreddits = f.read().splitlines()

        # Group by 'author'
        gb = df_all.groupby('author')

        # For each author, find the first mention of a subreddit and filter out later posts
        dfs = []
        for name, group in gb:
            group.sort_values(by='created_utc', inplace=True)
            for subreddit in subreddits:
                if subreddit in group['subreddit'].values:
                    first_mention_index = group[group['subreddit'] == subreddit].index.min()
                    group = group[group.index < first_mention_index]
                    break
            dfs.append(group)

        df_all = pd.concat(dfs)

    # Group by 'author' and join all 'text' for each author
    df_combined = df_all.groupby('author')['text'].apply('<s> '.join).reset_index()

    # Add 'label' column
    df_combined['label'] = 1 if 'hate' in directory else 0

    return df_combined

# longformers emedding portion
def embed_text(texts):
    embeddings = []
    for text in tqdm(texts):
        encoding = tokenizer.encode_plus(text, truncation=False, padding=False)

        if len(encoding["input_ids"]) > 4096:
            encoding["input_ids"] = encoding["input_ids"][-4096:]
            encoding["attention_mask"] = encoding["attention_mask"][-4096:]

        encoding = {key: torch.tensor(val).unsqueeze(0).to(DEV) for key, val in encoding.items()}

        global_attention_mask = torch.tensor([([0] * len(encoding["input_ids"][0]))]).to(DEV)  # make all zeros for local attention, all 1s for all global
        encoding["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            o = model(**encoding)

        sentence_embedding = o.last_hidden_state[:, 0].cpu().numpy().flatten()
        embeddings.append(sentence_embedding)

    return embeddings

def main():
    tqdm.pandas()

    # File paths
    hate_file_path = r'C:\Users\fujin\Downloads\test\test_json\hate'
    control_file_path = r'C:\Users\fujin\Downloads\test\test_json\control'
    subreddits_path = r'C:\Users\fujin\Downloads\hate_subreddits.txt'
    output_file_path = r'C:\Users\fujin\Downloads\hate_model_test_set.pt'

    # Initialize a list to hold all preprocessed dataframes
    df_list = []
    embeddings_list = []
    labels_list = []

    # Preprocess text from both directories and embed
    for directory in [hate_file_path, control_file_path]:
        df_preprocessed = preprocess_text(directory, subreddits_path)
        embeddings = embed_text(df_preprocessed['text'])
        df_preprocessed['embeddings'] = embeddings
        df_list.append(df_preprocessed)

        # Convert embeddings and labels to tensors and add to lists
        embeddings_list.extend([torch.tensor(e) for e in embeddings])
        labels_list.extend([torch.tensor(l) for l in df_preprocessed['label'].values])

    # Concatenate all preprocessed dataframes
    df_new = pd.concat(df_list, ignore_index=True)

    # Convert 'label' to integer
    df_new['label'] = df_new['label'].astype(int)

    # Drop the 'text' column
    df_new = df_new.drop(columns=['text'])

    # Convert lists of tensors to single tensors
    embeddings_tensor = torch.stack(embeddings_list)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    # Create a TensorDataset from embeddings and labels
    dataset = TensorDataset(embeddings_tensor, labels_tensor)

    # Save the TensorDataset
    torch.save(dataset, output_file_path)

    # Load the saved TensorDataset
    loaded_dataset = torch.load(output_file_path)
    print(loaded_dataset.tensors)
    print(loaded_dataset.tensors[0].shape)

    return df_new

if __name__ == "__main__":
    df_new = main()
    print(df_new)

