import json
import os
import zstandard
import glob
import io

def write_to_author_file(output_folder, author, data, data_type):
    with open(os.path.join(output_folder, f"{author}_{data_type}.json"), "a") as file:
        json.dump(data, file)
        file.write("\n")  # newline character so each line is a separate JSON object

def process_dump(path, values, output_folder):
    # Determines whether the file is a comment or a submission
    data_type = 'submissions' if 'RS' in os.path.basename(path) else 'comments'

    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)

    # Opens compressed file
    with open(path, 'rb') as fh:
        # Create a stream reader
        reader = dctx.stream_reader(fh)

        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        # For each line in the text stream
        for line in text_stream:
            # Remove trailing newline character
            line = line.rstrip('\n')

            try:
                # Convert the line to a dictionary
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Could not decode line as JSON: {line}")
                continue

            # If the 'author' key is in the dictionary and the author is in our list of authors
            if 'author' in data and data['author'] in values:
                print(f"Writing data for author {data['author']} to {data_type} file")  # Debug line
                # Write the data to the appropriate file
                write_to_author_file(output_folder, data['author'], data, data_type)

# Read the authors from a file
with open(r'C:\Users\fujin\Downloads\test\50k_authors.txt', 'r') as file:
    values = set(author.strip() for author in file.read().split(','))

# Set the paths to the input directories
input_folders = [r'C:\Users\fujin\Downloads\reddit_zst\reddit\comments',
                 r'C:\Users\fujin\Downloads\reddit_zst\reddit\submissions']  # add more folders as needed

# Set the path to the output folder
output_folder = r'C:\Users\fujin\Downloads\test\test_json\control'

# For each input directory in the list
for input_folder in input_folders:
    # For each file in the current input directory
    for path in glob.glob(os.path.join(input_folder, '*.zst')):
        print(f"Processing file {path}")  # Debug line
        process_dump(path, values, output_folder)
