#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import json
import token
from swifter import swifter

warnings.filterwarnings('ignore')

# General Packages
import numpy as np
import pandas as pd
import os
import time
from glob import glob

# Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['svg.fonttype'] = 'none'
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Iterators
from collections import Counter
from itertools import islice
from operator import itemgetter
from tqdm import tqdm

tqdm.pandas()

# Parallelization
import swifter
from multiprocessing import Process

# Text
import re
import string
from textblob import TextBlob
from nrclex import NRCLex
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, MWETokenizer
from nltk.stem import porter, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer

# Statistical Analysis
from scipy import stats

# In[2]:

#import nltk libraries

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

HATE_FOLDER = "/data/jfu/reddit/hate"
CONTROL_FOLDER = "/data/jfu/reddit/new_control2"
#HATE_FOLDER = r"C:\Users\fujin\Downloads\test\test_json\hate_test"
#CONTROL_FOLDER = r"C:\Users\fujin\Downloads\test\test_json\control_test"

if __name__ == "__main__":

    # In[3]:

    start_time = time.time()


    # In[4]:

    def run_cpu_tasks_in_parallel(tasks):
        running_tasks = [Process(target=task) for task in tasks]
        for running_task in running_tasks:
            running_task.start()
        for running_task in running_tasks:
            running_task.join()


    ## Load & Clean Data

    # In[5]:

    step_start = time.time()

    # In[6]:

    # Fetch a list of all files to read
    hate_json = glob(HATE_FOLDER + '/*.json')
    control_json = glob(CONTROL_FOLDER + '/*.json')

    # In[7]:

    # Series to store the tokens from all iterations
    clean_tokens_avax = pd.Series()
    clean_tokens_random = pd.Series()

    # In[8]:

    # Set stopwords
    stop = stopwords.words('english')
    stop += ['nan', '.', ',', ':', '...', '!"', '?"', "'", '"', ' - ', ' — ', ',"', '."', '!', ';', '’', 's', 'one',
             'get', 'it', 'di', 'para',
             '.\'"', '[', ']', '—', ".\'", 'ok', 'okay', 'yeah', 'ya', 'stuff', ' 000 ', '“', '”', ' ’', 'im', 'el',
             'les', 'le',
             't', 'was', 'wa', '?', 'like', 'go', "'s", 'i', ' I ', " ? ", "s", " t ", "ve", "re", 'guy', 'de', 'la',
             'que', 'en', 'u', 'los', 'would', 'dont', 'the',
             '0', '00', '000', 'à', 'a', 'e', 'w', 'es', 'por', 'also', 'one', 'two', 'three',
             'se', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
             '17', '18', '19', '20',
             'क', 'ह', 'nan']

    # Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Function to clean stopwords
    def remove_stopwords(sentence):
        clean = ' '.join([re.sub(r'[^\w\s]', '', word.strip()) for word in sentence.split() if re.sub(r'[^\w\s]', '', word.strip()) not in stop])
        return ' '.join([lemmatizer.lemmatize(word) for word in clean.split() if len(word) > 1])

    # JAMES SECTION
    def process_json(json_list, filename, is_hate_data=False):
        all_tokens = []

        # Define the list of specific subreddits
        specific_subreddits = [
            "asktrp", "Braincels", "CCJ2", "CoonTown", "CringeAnarchy",
            "Delraymisfits", "dolan", "fatpeoplehate", "FuckYou", "frenWorld",
            "GenderCritical", "GreatApes", "honkler", "ImGoingToHellForThis",
            "Incels", "MGTOW", "milliondollarextreme", "Neofag", "opieandanthony",
            "sjwhate", "TheRedPill", "TrollGC", "TruFemcels", "Trolling", "WhiteRights"
        ]

        for json_path in json_list:
            df = pd.read_json(json_path, lines=True)
            df = df.sample(frac=0.1)

            df.sort_values(by='created_utc', inplace=True)

            # NEW IMPLEMENTATION check if the length of the file is 0, disregard the file
            if df.size == 0:
                #print("empty dataframe")
                continue
            elif df.size > 0:
                print("dataframe not empty")

            if is_hate_data:
                try:
                    # try and except, inspect the data, find the failing file
                    # Find the first occurrence of the specific subreddits
                    first_occurrences = df[df['subreddit'].isin(specific_subreddits)].index
                    if not first_occurrences.empty:
                        earliest_occurrence = first_occurrences.min()
                        df = df[df.index < earliest_occurrence]
                except KeyError:
                    print(json_path, " has no data")
                    continue

            # Check the JSON data to determine if it is a submission or a comment
            if 'selftext' in df.columns:
                df['token'] = df['selftext'].astype(str) + ' ' + df['title'].astype(str)
                print(json_path, "string has data")
            elif 'body' in df.columns:
                df['token'] = df['body'].astype(str)
                print(json_path, "string has data")
            else:
                # no data.... new section
                print(json_path, "string has no data")
            if 'token' not in df.columns:
                df['token'] = ''

            # Process tokens
            # pre-process tokens to lower case
            df['token'] = df['token'].str.lower()
            iteration_tokens = df['token'].apply(remove_stopwords)
            all_tokens.extend(iteration_tokens)

        # Convert the list to a DataFrame
        clean_tokens_df = pd.DataFrame(all_tokens)

        # Save to CSV
        clean_tokens_df.to_csv(filename, header=False, index=False)
        # Convert the list to a DataFrame
        clean_tokens_df = pd.DataFrame(all_tokens)

        # Save to CSV
        clean_tokens_df.to_csv(filename, header=False, index=False)


    # Process 'hate_json' and 'control_json'
    try:
        clean_tokens_avax = pd.read_csv('/data/jfu/reddit/clean_tokens_avax.csv', header=None).iloc[:, 0]
        print("Loaded tokens from disk.")
    except FileNotFoundError:
        print("File 'clean_tokens_avax.csv' not found. Creating a new one...")
        process_json(hate_json, '/data/jfu/reddit/clean_tokens_avax.csv', is_hate_data=True)

    try:
        clean_tokens_random = pd.read_csv('/data/jfu/reddit/clean_tokens_random.csv', header=None).iloc[:, 0]
        print("Loaded tokens from disk.")
    except FileNotFoundError:
        print("File 'clean_tokens_random.csv' not found. Creating a new one...")
        process_json(control_json, '/data/jfu/reddit/clean_tokens_random.csv', is_hate_data=False)


    # In[11]:

    step_end = time.time()
    print()
    print(f'Processing Time - Load & Clean Data: {(step_end - step_start) / 60:.0f} minutes')
    print(f'Total Time - Load & Clean Data: {(step_end - start_time) / 60:.0f} minutes')
    print()

    ## Word Frequencies

    # In[13]:

    step_start = time.time()

    #MY CHANGES - top hate and control words and df_merged

    # In[14]:
    try:
        df_merged = pd.read_csv('/data/jfu/reddit/df_merged.csv', index_col=0)
        print("df_merged read from disk")

    except:
        def take(n, iterable):
            return list(islice(iterable, n))

        #drop nans
        clean_tokens_avax = clean_tokens_avax.dropna()
        clean_tokens_random = clean_tokens_random.dropna()

        clean_tokens_avax = clean_tokens_avax.fillna("")
        clean_tokens_random = clean_tokens_random.fillna("")

        WC_avax = Counter(word for post in clean_tokens_avax for word in post.split())
        WC_avax_sorted = {k: v for k, v in sorted(WC_avax.items(), key=lambda item: item[1], reverse=True)}
        top_words_avax = take(15, WC_avax_sorted.items())

        WC_random = Counter(word for post in clean_tokens_random for word in post.split())
        WC_random_sorted = {k: v for k, v in sorted(WC_random.items(), key=lambda item: item[1], reverse=True)}
        top_words_random = take(15, WC_random_sorted.items())

        print('Top hate words: ')
        print(top_words_avax)
        print()
        print('Top control words: ')
        print(top_words_random)

        df_avax_words = pd.DataFrame.from_dict(WC_avax_sorted, orient='index', columns=['count_hate'])
        df_avax_words['percentage_hate'] = df_avax_words['count_hate'] / df_avax_words['count_hate'].sum()

        df_random_words = pd.DataFrame.from_dict(WC_random_sorted, orient='index', columns=['count_random'])
        df_random_words['percentage_random'] = df_random_words['count_random'] / df_random_words['count_random'].sum()

        df_merged = pd.merge(df_avax_words, df_random_words, how='outer', left_index=True, right_index=True)
        df_merged = df_merged.fillna(0)
        df_merged['count_hate'] = df_merged['count_hate'].astype(int)
        df_merged['count_random'] = df_merged['count_random'].astype(int)
        df_merged['count_total'] = df_merged['count_hate'] + df_merged['count_random']
        df_merged.sort_values('count_total', ascending=False, inplace=True)

        df_merged['ratio'] = df_merged['percentage_hate'] / df_merged['percentage_random']

        df_merged.to_csv('/data/jfu/reddit/df_merged.csv', index=True)

    # Filter top words
    top50_avax = df_merged.nlargest(5, 'percentage_hate')
    top50_random = df_merged.nlargest(5, 'percentage_random')
    top50_overall = df_merged.nlargest(5, 'count_total')  # Corrected to get top 5

    df_top_words_abs = pd.concat([top50_avax, top50_random, top50_overall])
    df_top_words_abs = df_top_words_abs[~df_top_words_abs.index.duplicated(keep='first')]

    # Remove those with a small ratio, as they are not discriminative
    # df_top_words_abs = df_top_words_abs[(df_top_words_abs.ratio > 2.0) | (df_top_words_abs.ratio < 0.5)]

    # In[141]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Scatterplot of top 100 words by overall count
    sns.scatterplot(x=df_merged['percentage_random'], y=df_merged['percentage_hate'], 
                    color='lightgray', alpha=0.20, s=75, linewidth=0)

    # Add labels to words
    for word, row in df_top_words_abs.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Avax Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='bold', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='bold', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Absolute_Scatter.png', format='png')
    '''

    # In[20]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Heatmap of word frequencies
    sns.histplot(df_merged, x="percentage_random", y="percentage_hate", hue='count_total', 
                 bins=250, palette='magma_r', legend=False, ax=ax,
                 hue_norm=matplotlib.colors.LogNorm())

    # Add labels to words
    for word, row in df_top_words_abs.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Avax Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='bold', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='bold', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Absolute_Heatmap.png', format='png')
    '''

    # WORKING UP TO HERE

    print('Word Frequencies with absolute scale completed!')

    # #### Relative Frequency

    # In[142]:

    df_merged = df_merged.fillna(0)

    # Keep only those with at least some small percentage in each
    df_filtered_1 = df_merged[(df_merged.percentage_hate > 10e-2) & (df_merged.percentage_hate <= 10e-1) & (
                df_merged.percentage_random > 10e-7)]
    df_filtered_2 = df_merged[(df_merged.percentage_hate > 10e-3) & (df_merged.percentage_hate <= 10e-2) & (
                df_merged.percentage_random > 10e-7)]
    df_filtered_3 = df_merged[(df_merged.percentage_hate > 10e-4) & (df_merged.percentage_hate <= 10e-3) & (
                df_merged.percentage_random > 10e-7)]
    df_filtered_4 = df_merged[(df_merged.percentage_hate > 10e-5) & (df_merged.percentage_hate <= 10e-4) & (
                df_merged.percentage_random > 10e-7)]
    df_filtered_5 = df_merged[(df_merged.percentage_hate > 10e-6) & (df_merged.percentage_hate <= 10e-5) & (
                df_merged.percentage_random > 10e-7)]
    df_filtered_6 = df_merged[(df_merged.percentage_hate > 10e-7) & (df_merged.percentage_hate <= 10e-6) & (
                df_merged.percentage_random > 10e-7)]

    # Filter top words per class and globally to use for plotting
    ratio_avax_1 = df_filtered_1.nlargest(4, 'ratio')
    ratio_random_1 = df_filtered_1.nsmallest(4, 'ratio')

    ratio_avax_2 = df_filtered_2.nlargest(4, 'ratio')
    ratio_random_2 = df_filtered_2.nsmallest(4, 'ratio')

    ratio_avax_3 = df_filtered_3.nlargest(4, 'ratio')
    ratio_random_3 = df_filtered_3.nsmallest(4, 'ratio')

    ratio_avax_4 = df_filtered_4.nlargest(0, 'ratio')
    ratio_random_4 = df_filtered_4.nsmallest(4, 'ratio')

    ratio_avax_5 = df_filtered_5.nlargest(4, 'ratio')
    ratio_random_5 = df_filtered_5.nsmallest(4, 'ratio')

    ratio_avax_6 = df_filtered_6.nlargest(4, 'ratio')
    ratio_random_6 = df_filtered_6.nsmallest(4, 'ratio')

    # Merge and deduplicate
    df_top_words_rel1 = pd.concat([ratio_avax_1, ratio_random_1,
                                   ratio_avax_2, ratio_random_2,
                                   ratio_avax_3, ratio_random_3,
                                   ratio_avax_4, ratio_random_4,
                                   ratio_avax_5, ratio_random_5,
                                   ratio_avax_6, ratio_random_6])
    df_top_words_rel1 = df_top_words_rel1[~df_top_words_rel1.index.duplicated(keep='first')]

    # Keep only those with at least some small percentage in each
    df_filtered_1 = df_merged[(df_merged.percentage_random > 10e-2) & (df_merged.percentage_random <= 10e-1) & (
                df_merged.percentage_hate > 10e-7)]
    df_filtered_2 = df_merged[(df_merged.percentage_random > 10e-3) & (df_merged.percentage_random <= 10e-2) & (
                df_merged.percentage_hate > 10e-7)]
    df_filtered_3 = df_merged[(df_merged.percentage_random > 10e-4) & (df_merged.percentage_random <= 10e-3) & (
                df_merged.percentage_hate > 10e-7)]
    df_filtered_4 = df_merged[(df_merged.percentage_random > 10e-5) & (df_merged.percentage_random <= 10e-4) & (
                df_merged.percentage_hate > 10e-7)]
    df_filtered_5 = df_merged[(df_merged.percentage_random > 10e-6) & (df_merged.percentage_random <= 10e-5) & (
                df_merged.percentage_hate > 10e-7)]
    df_filtered_6 = df_merged[(df_merged.percentage_random > 10e-7) & (df_merged.percentage_random <= 10e-6) & (
                df_merged.percentage_hate > 10e-7)]

    # Filter top words per class and globally to use for plotting
    ratio_avax_1 = df_filtered_1.nlargest(4, 'ratio')
    ratio_random_1 = df_filtered_1.nsmallest(0, 'ratio')

    ratio_avax_2 = df_filtered_2.nlargest(4, 'ratio')
    ratio_random_2 = df_filtered_2.nsmallest(0, 'ratio')

    ratio_avax_3 = df_filtered_3.nlargest(4, 'ratio')
    ratio_random_3 = df_filtered_3.nsmallest(0, 'ratio')

    ratio_avax_4 = df_filtered_4.nlargest(4, 'ratio')
    ratio_random_4 = df_filtered_4.nsmallest(0, 'ratio')

    ratio_avax_5 = df_filtered_5.nlargest(4, 'ratio')
    ratio_random_5 = df_filtered_5.nsmallest(0, 'ratio')

    ratio_avax_6 = df_filtered_6.nlargest(4, 'ratio')
    ratio_random_6 = df_filtered_6.nsmallest(0, 'ratio')

    # Merge and deduplicate
    df_top_words_rel2 = pd.concat([ratio_avax_1, ratio_random_1,
                                   ratio_avax_2, ratio_random_2,
                                   ratio_avax_3, ratio_random_3,
                                   ratio_avax_4, ratio_random_4,
                                   ratio_avax_5, ratio_random_5,
                                   ratio_avax_6, ratio_random_6])
    df_top_words_rel2 = df_top_words_rel2[~df_top_words_rel2.index.duplicated(keep='first')]

    # Merge the selected words from both classes
    df_top_words_rel = pd.concat([df_top_words_rel1, df_top_words_rel2])
    df_top_words_rel = df_top_words_rel[~df_top_words_rel.index.duplicated(keep='first')]

    # In[143]:

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))

    # Scatterplot of top 100 words by overall count
    sns.scatterplot(x=df_merged['percentage_random'], y=df_merged['percentage_hate'],
                    color='lightgray', alpha=0.20, s=75, linewidth=0)

    # Add labels to words
    for word, row in df_top_words_rel.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black',
                alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=20)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Hate Users x Random Users', pad=20,
              fontdict={'fontsize': 30,
                        'fontweight': 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=25, weight='normal', labelpad=20)
    plt.ylabel('Word Frequency in Hate Posts', fontsize=25, weight='normal', labelpad=20)

    # Display without grid
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    fig.savefig('/data/jfu/reddit/Word_Frequencies_Ratio_Scatter.png', format='png')

    # In[42]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Heatmap of word frequencies
    sns.histplot(df_merged, x="percentage_random", y="percentage_hate", hue='count_total', 
                 bins=100, palette='magma_r', legend=False, ax=ax,
                 hue_norm=matplotlib.colors.LogNorm())

    # Add labels to words
    for word, row in df_top_words_rel.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Hate Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='bold', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='bold', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Ratio_Heatmap.png', format='png')
    '''
    print('Word Frequencies with relative scale completed!')

    # #### Metric

    # In[144]:

    # Define a filtering metric
    df_merged['metric'] = (abs(np.log2(df_merged['ratio'])) ** 3) * (df_merged['percentage_hate'] ** 1) * (
                df_merged['percentage_random'] ** 1)  # * (df_merged['count_total']**2)

    # Filter top words per class and globally to use for plotting
    metric_top_words = df_merged.nlargest(5, 'metric')

    # Merge and deduplicate
    df_top_words_met = pd.concat([metric_top_words])
    df_top_words_met = df_top_words_met[~df_top_words_met.index.duplicated(keep='first')]

    # In[145]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Scatterplot of top 100 words by overall count
    sns.scatterplot(x=df_merged['percentage_random'], y=df_merged['percentage_hate'], 
                    color='lightgray', alpha=0.20, s=75, linewidth=0)

    # Add labels to words
    for word, row in df_top_words_met.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Avax Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='bold', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='bold', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Metric_Scatter.png', format='png')
    '''

    # In[ ]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Heatmap of word frequencies
    sns.histplot(df_merged, x="percentage_random", y="percentage_hate", hue='count_total', 
                 bins=100, palette='magma_r', legend=False, ax=ax,
                 hue_norm=matplotlib.colors.LogNorm())

    # Add labels to words
    for word, row in df_top_words_met.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Avax Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='bold', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='bold', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Metric_Heatmap.png', format='png')
    '''
    print('Word Frequencies with metric scale completed!')

    # #### All Rules

    # In[146]:

    # Using both rules
    df_top_words = pd.concat([df_top_words_abs, df_top_words_rel, df_top_words_met])
    df_top_words = df_top_words[~df_top_words.index.duplicated(keep='first')]

    # In[148]:

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))

    # Scatterplot of top 100 words by overall count
    sns.scatterplot(x=df_merged['percentage_random'], y=df_merged['percentage_hate'],
                    color='lightgray', alpha=0.20, s=75, linewidth=0)

    # Add labels to words
    for word, row in df_top_words.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black',
                alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=20)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Hate x Not Hate', pad=20,
              fontdict={'fontsize': 30,
                        'fontweight': 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=25, weight='normal', labelpad=20)
    plt.ylabel('Word Frequency in Hate Posts', fontsize=25, weight='normal', labelpad=20)

    # Display without grid
    plt.grid(False)
    plt.tight_layout()
    # plt.show()
    fig.savefig('/data/jfu/reddit/Word_Frequencies_Merged_Scatter.png', format='png')

    # In[ ]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(15,15))

    # Heatmap of word frequencies
    sns.histplot(df_merged, x="percentage_random", y="percentage_hate", hue='count_total', 
                 bins=100, palette='magma_r', legend=False, ax=ax,
                 hue_norm=matplotlib.colors.LogNorm())

    # Add labels to words
    for word, row in df_top_words.iterrows():
        ax.text(x=row['percentage_random'], y=row['percentage_hate'], s=word, fontdict={'fontsize': 16}, color='black', alpha=0.75)

    # Format axis
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='both', labelsize=14)
    y_max = df_merged['percentage_hate'].max()
    y_min = df_merged[df_merged['percentage_hate'] > 0]['percentage_hate'].min()
    x_max = df_merged['percentage_random'].max()
    x_min = df_merged[df_merged['percentage_random'] > 0]['percentage_random'].min()
    scale_max = max(y_max, x_max) * 1.1
    scale_min = min(y_min, x_min) * 15
    ax.set_xlim(scale_min, scale_max)
    ax.set_ylim(scale_min, scale_max)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="darkslategray", alpha=0.5, lw=1)

    # Title and labels
    plt.title('Word Frequencies: Avax Users x Random Users', pad=20, 
              fontdict={'fontsize': 20,
                        'fontweight' : 'bold'})
    plt.xlabel('Word Frequency in Random Posts', fontsize=16, weight='normal', labelpad=20)
    plt.ylabel('Word Frequency in Avax Posts', fontsize=16, weight='normal', labelpad=20)

    # Display without grid
    plt.grid(False)
    #plt.show()
    fig.savefig('Word_Frequencies_Merged_Heatmap.png', format='png')
    '''
    print('Word Frequencies with merged scale completed!')

    # In[27]:

    # del WC_avax_sorted, WC_random_sorted, df_avax_words, df_random_words, df_merged, top50_avax, top50_random, top50_overall, df_top_words_abs, df_top_words_rel, metric_top_words, df_top_words_met, df_top_words

    step_end = time.time()
    print()
    print(f'Processing Time - Word Frequencies: {(step_end - step_start) / 60:.0f} minutes')
    print(f'Total Time - Word Frequencies: {(step_end - start_time) / 60:.0f} minutes')
    print()

    # ## Emotions

    # In[20]:

    step_start = time.time()


    # In[21]:

    # Function to extract affects in parallel
    def get_affect(tweet):
        affect_dict = NRCLex(tweet).affect_frequencies
        if 'anticipation' not in affect_dict:
            affect_dict['anticipation'] = np.NaN
        return affect_dict


    # In[22]:

    # Extract affects in parallel
    df_sentiments_avax = pd.DataFrame(clean_tokens_avax)
    df_sentiments_avax.columns = ['token']
    df_sentiments_avax['affect'] = df_sentiments_avax['token'].swifter.progress_bar(True).apply(get_affect)
    df_emotions_avax = df_sentiments_avax.copy()
    df_sentiments_avax.to_csv("/data/jfu/reddit/df_sentiments_avax.csv", index=True)
    df_emotions_avax.to_csv("/data/jfu/reddit/df_emotions_avax.csv", index=True)
    print('Avax affects parsed from corpus.')

    df_sentiments_random = pd.DataFrame(clean_tokens_random)
    df_sentiments_random.columns = ['token']
    df_sentiments_random['affect'] = df_sentiments_random['token'].swifter.progress_bar(True).apply(get_affect)
    df_emotions_random = df_sentiments_random.copy()
    df_sentiments_random.to_csv("/data/jfu/reddit/df_sentiments_random.csv", index=True)
    df_emotions_random.to_csv("/data/jfu/reddit/df_emotions_random.csv", index=True)
    print('Random affects parsed from corpus.')

    # In[23]:

    # Convert the affects dictionary into columns
    df_sentiments_avax['fear'] = df_sentiments_avax['affect'].apply(lambda row: row['fear'])
    df_sentiments_avax['anger'] = df_sentiments_avax['affect'].apply(lambda row: row['anger'])
    df_sentiments_avax['anticipation'] = df_sentiments_avax['affect'].apply(lambda row: row['anticipation'])
    df_sentiments_avax['trust'] = df_sentiments_avax['affect'].apply(lambda row: row['trust'])
    df_sentiments_avax['surprise'] = df_sentiments_avax['affect'].apply(lambda row: row['surprise'])
    # df_sentiments_avax['positive'] = df_sentiments_avax['affect'].apply(lambda row: row['positive'])
    # df_sentiments_avax['negative'] = df_sentiments_avax['affect'].apply(lambda row: row['negative'])
    df_sentiments_avax['sadness'] = df_sentiments_avax['affect'].apply(lambda row: row['sadness'])
    df_sentiments_avax['disgust'] = df_sentiments_avax['affect'].apply(lambda row: row['disgust'])
    df_sentiments_avax['joy'] = df_sentiments_avax['affect'].apply(lambda row: row['joy'])
    df_sentiments_avax.drop(['token', 'affect'], axis=1, inplace=True)

    df_sentiments_random['fear'] = df_sentiments_random['affect'].apply(lambda row: row['fear'])
    df_sentiments_random['anger'] = df_sentiments_random['affect'].apply(lambda row: row['anger'])
    df_sentiments_random['anticipation'] = df_sentiments_random['affect'].apply(lambda row: row['anticipation'])
    df_sentiments_random['trust'] = df_sentiments_random['affect'].apply(lambda row: row['trust'])
    df_sentiments_random['surprise'] = df_sentiments_random['affect'].apply(lambda row: row['surprise'])
    # df_sentiments_random['positive'] = df_sentiments_random['affect'].apply(lambda row: row['positive'])
    # df_sentiments_random['negative'] = df_sentiments_random['affect'].apply(lambda row: row['negative'])
    df_sentiments_random['sadness'] = df_sentiments_random['affect'].apply(lambda row: row['sadness'])
    df_sentiments_random['disgust'] = df_sentiments_random['affect'].apply(lambda row: row['disgust'])
    df_sentiments_random['joy'] = df_sentiments_random['affect'].apply(lambda row: row['joy'])
    df_sentiments_random.drop(['token', 'affect'], axis=1, inplace=True)

    # In[24]:

    # Generate dataframe with ['emotion', 'percentage', 'class']
    avax_sentiment_scores = pd.DataFrame(df_sentiments_avax.sum(axis=0))
    avax_sentiment_scores['class'] = 'Hate'
    avax_sentiment_scores.reset_index(inplace=True, drop=False)
    avax_sentiment_scores.columns = ['emotion', 'percentage', 'class']
    avax_sentiment_scores['percentage'] = avax_sentiment_scores['percentage'] / avax_sentiment_scores[
        'percentage'].sum()

    random_sentiment_scores = pd.DataFrame(df_sentiments_random.sum(axis=0))
    random_sentiment_scores['class'] = 'Not Hate'
    random_sentiment_scores.reset_index(inplace=True, drop=False)
    random_sentiment_scores.columns = ['emotion', 'percentage', 'class']
    random_sentiment_scores['percentage'] = random_sentiment_scores['percentage'] / random_sentiment_scores[
        'percentage'].sum()

    # In[25]:

    # Merge class dataframes into a global dataframe
    df_sentiment_scores = pd.concat([avax_sentiment_scores, random_sentiment_scores])
    df_sentiment_scores.sort_values('emotion', inplace=True)
    df_sentiment_scores.to_csv('/data/jfu/reddit/df_sentiment_scores.csv', index=True)

    # In[26]:

    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(16,9), sharey=True)

    # Plot emotions
    sns.barplot(data=df_sentiment_scores, x='emotion', y='percentage', hue='class', 
                palette={'Hate':'gold', 'Not Hate':'cyan'}, alpha=0.8, saturation=0.65)

    # Set axis to percentage
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Title and labels
    plt.suptitle('Expressed Sentiments', size=34, weight='bold')
    plt.xticks(fontsize=25, rotation=30)
    plt.yticks(fontsize=20)
    plt.xlabel('Sentiment', fontsize=30, weight='normal', labelpad=20)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)

    # Display
    plt.legend(fontsize=25)
    plt.tight_layout()
    #plt.show()
    fig.savefig('Sentiments.png', format='png')
    '''

    # In[27]:

    # Repeat for emotions
    df_emotions_avax['positive'] = df_emotions_avax['affect'].apply(lambda row: row['positive'])
    df_emotions_avax['negative'] = df_emotions_avax['affect'].apply(lambda row: row['negative'])
    df_emotions_avax.drop(['token', 'affect'], axis=1, inplace=True)

    df_emotions_random['positive'] = df_emotions_random['affect'].apply(lambda row: row['positive'])
    df_emotions_random['negative'] = df_emotions_random['affect'].apply(lambda row: row['negative'])
    df_emotions_random.drop(['token', 'affect'], axis=1, inplace=True)

    # Generate dataframe with ['emotion', 'percentage', 'class']
    avax_emotion_scores = pd.DataFrame(df_emotions_avax.sum(axis=0))
    avax_emotion_scores['class'] = 'Hate'
    avax_emotion_scores.reset_index(inplace=True, drop=False)
    avax_emotion_scores.columns = ['emotion', 'percentage', 'class']
    avax_emotion_scores['percentage'] = avax_emotion_scores['percentage'] / avax_emotion_scores['percentage'].sum()

    random_emotion_scores = pd.DataFrame(df_emotions_random.sum(axis=0))
    random_emotion_scores['class'] = 'Not Hate'
    random_emotion_scores.reset_index(inplace=True, drop=False)
    random_emotion_scores.columns = ['emotion', 'percentage', 'class']
    random_emotion_scores['percentage'] = random_emotion_scores['percentage'] / random_emotion_scores[
        'percentage'].sum()

    # Merge class dataframes into a global dataframe
    df_emotion_scores = pd.concat([avax_emotion_scores, random_emotion_scores])
    df_emotion_scores.sort_values('emotion', inplace=True)
    df_emotion_scores.to_csv('/data/jfu/reddit/df_emotion_scores.csv', index=True)

    # In[98]
    '''
    # Create figure
    fig, ax = plt.subplots(figsize=(16,9), sharey=True)

    # Plot emotions
    sns.barplot(data=df_emotion_scores, x='emotion', y='percentage', hue='class', 
                palette={'Hate':'gold', 'Not Hate':'cyan'}, alpha=0.8, saturation=0.65)

    # Set axis to percentage
    ax.yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Title and labels
    plt.suptitle('Expressed Emotions', size=34, weight='bold')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=20)
    plt.xlabel('Emotion', fontsize=30, weight='normal', labelpad=20)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)

    # Display
    plt.legend(fontsize=25)
    plt.tight_layout()
    #plt.show()
    fig.savefig('Emotions.png', format='png') 
    '''

    # Two-in-One plot

    fig, ax = plt.subplots(ncols=2, figsize=(16, 9), gridspec_kw={'width_ratios': [1, 4]})  # , sharey=True)

    plt.suptitle("Affects: Emotions and Sentiments", size=30, weight='bold')

    fig.sca(ax[0])
    sns.barplot(data=df_emotion_scores, x='emotion', y='percentage', hue='class',
                palette={'Hate': 'gold', 'Not Hate': 'cyan'}, alpha=0.8, saturation=0.65)
    ax[0].yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(fontsize=25, rotation=30)
    plt.yticks(fontsize=20)
    plt.xlabel('Emotion', fontsize=30, weight='normal', labelpad=20)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)
    plt.legend('')

    fig.sca(ax[1])
    sns.barplot(data=df_sentiment_scores, x='emotion', y='percentage', hue='class',
                palette={'Hate': 'gold', 'Not Hate': 'cyan'}, alpha=0.8, saturation=0.65)
    ax[1].yaxis.set_major_locator(mtick.MultipleLocator(0.1))
    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(fontsize=25, rotation=30)
    plt.yticks(fontsize=20)
    plt.xlabel('Sentiment', fontsize=30, weight='normal', labelpad=20)
    plt.ylabel('', fontsize=30, weight='normal', labelpad=20)
    plt.legend(fontsize=25)

    fig.align_xlabels(ax)
    fig.tight_layout()
    plt.show()
    fig.savefig('/data/jfu/reddit/Affects.png', format='png')

    # # ### Mann-Whitney U Test

    # # In[178]:

    # # Sentiments
    # mw_fear = stats.mannwhitneyu(df_sentiments_avax['fear'].dropna(), df_sentiments_random['fear'].dropna())
    # mw_anger = stats.mannwhitneyu(df_sentiments_avax['anger'].dropna(), df_sentiments_random['anger'].dropna())
    # mw_anticipation = stats.mannwhitneyu(df_sentiments_avax['anticipation'].dropna(), df_sentiments_random['anticipation'].dropna())
    # mw_trust = stats.mannwhitneyu(df_sentiments_avax['trust'].dropna(), df_sentiments_random['trust'].dropna())
    # mw_surprise = stats.mannwhitneyu(df_sentiments_avax['surprise'].dropna(), df_sentiments_random['surprise'].dropna())
    # mw_sadness = stats.mannwhitneyu(df_sentiments_avax['sadness'].dropna(), df_sentiments_random['sadness'].dropna())
    # mw_disgust = stats.mannwhitneyu(df_sentiments_avax['disgust'].dropna(), df_sentiments_random['disgust'].dropna())
    # mw_joy = stats.mannwhitneyu(df_sentiments_avax['joy'].dropna(), df_sentiments_random['joy'].dropna())

    # # Emotions
    # mw_positive = stats.mannwhitneyu(df_emotions_avax['positive'].dropna(), df_emotions_random['positive'].dropna())
    # mw_negative = stats.mannwhitneyu(df_emotions_avax['negative'].dropna(), df_emotions_random['negative'].dropna())

    # # In[180]:

    # # DataFrame to store t-tests
    # df_mannwhitneyu = pd.DataFrame(columns=['feature', 'statistic', 'pvalue'])

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'fear',
    #                               'statistic':mw_fear.statistic,
    #                               'pvalue':mw_fear.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'anger',
    #                               'statistic':mw_anger.statistic,
    #                               'pvalue':mw_anger.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'anticipation',
    #                               'statistic':mw_anticipation.statistic,
    #                               'pvalue':mw_anticipation.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'trust',
    #                               'statistic':mw_trust.statistic,
    #                               'pvalue':mw_trust.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'surprise',
    #                               'statistic':mw_surprise.statistic,
    #                               'pvalue':mw_surprise.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'sadness',
    #                               'statistic':mw_sadness.statistic,
    #                               'pvalue':mw_sadness.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'disgust',
    #                               'statistic':mw_disgust.statistic,
    #                               'pvalue':mw_disgust.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'joy',
    #                               'statistic':mw_joy.statistic,
    #                               'pvalue':mw_joy.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'positive',
    #                               'statistic':mw_positive.statistic,
    #                               'pvalue':mw_positive.pvalue},
    #                             ignore_index=True)

    # # Add to DataFrame
    # df_mannwhitneyu = df_mannwhitneyu.append({'feature':'negative',
    #                               'statistic':mw_negative.statistic,
    #                               'pvalue':mw_negative.pvalue},
    #                             ignore_index=True)

    # # In[181]:

    # # Save results to csv
    # df_mannwhitneyu.to_csv('mannwhitneyu_test_results_emotions.csv', index=False)

    # #del df_sentiments_avax, df_sentiments_random, avax_sentiment_scores, random_sentiment_scores
    # #del df_emotions_avax, df_emotions_random, avax_emotion_scores, random_emotion_scores

    # # In[99]

    # step_end = time.time()
    # print()
    # print(f'Processing Time - Emotions: {(step_end-step_start)/60:.0f} minutes')
    # print(f'Total Time - Emotions: {(step_end-start_time)/60:.0f} minutes')
    # print()

    # # ## TF-IDF

    # # In[28]:

    # '''
    # step_start = time.time()

    # # In[29]:

    # # Get TF-IDF values
    # vectorizer_avax = TfidfVectorizer()
    # vectorizer_random = TfidfVectorizer()

    # tf_idf_avax = vectorizer_avax.fit_transform(clean_tokens_avax.values)
    # tf_idf_random = vectorizer_random.fit_transform(clean_tokens_random.values)

    # feature_names_avax = vectorizer_avax.get_feature_names()
    # feature_names_random = vectorizer_random.get_feature_names()

    # # In[30]:

    # # Generate DataFrames
    # df_tf_idf_avax = pd.DataFrame()
    # df_tf_idf_avax['Feature Names'] = feature_names_avax
    # df_tf_idf_avax['Avg TF-IDF'] = np.ravel(tf_idf_avax.mean(axis=0))
    # df_tf_idf_avax.sort_values('Avg TF-IDF', ascending=False, inplace=True)

    # df_tf_idf_random = pd.DataFrame()
    # df_tf_idf_random['Feature Names'] = feature_names_random
    # df_tf_idf_random['Avg TF-IDF'] = np.ravel(tf_idf_random.mean(axis=0))
    # df_tf_idf_random.sort_values('Avg TF-IDF', ascending=False, inplace=True)

    # # In[31]:

    # # Merge top 15 words from each group
    # top_15_avax = df_tf_idf_avax[:20]
    # top_15_random = df_tf_idf_random[:20]
    # top_words_both = np.unique(list(top_15_avax['Feature Names'].values) + list(top_15_random['Feature Names'].values))

    # top_30_avax = df_tf_idf_avax[df_tf_idf_avax['Feature Names'].isin(top_words_both)]
    # top_30_avax['class'] = 'avax'
    # top_30_random = df_tf_idf_random[df_tf_idf_random['Feature Names'].isin(top_words_both)]
    # top_30_random['class'] = 'random'

    # top_TF_IDF_words = pd.concat((top_30_avax, top_30_random))

    # # In[32]:

    # # Create figure
    # fig, ax = plt.subplots(figsize=(15,15))

    # # Plot emotions
    # sns.barplot(data=top_TF_IDF_words, x='Avg TF-IDF', y='Feature Names', hue='class',
    #             palette={'avax':'gold', 'random':'cyan'}, alpha=0.8, saturation=0.65)

    # # Title and labels
    # plt.suptitle('Top TF-IDF Words: Avax Users x Random Users', size=20, weight='bold')
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.xlabel('TF-IDF', fontsize=16, weight='bold', labelpad=20)
    # plt.ylabel('')

    # # Display
    # plt.legend(fontsize=16)
    # #plt.show()
    # fig.savefig('TF_IDF.png', format='png')

    # # In[33]:

    # step_end = time.time()
    # print()
    # print(f'Processing Time - TF-IDF: {(step_end-step_start)/60:.0f} minutes')
    # print(f'Total Time - TF-IDF: {(step_end-start_time)/60:.0f} minutes')
    # print()
    # '''
