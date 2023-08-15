import zstandard
import os
import json
from tqdm import tqdm

# Directory path containing the ZST files
zst_directory_path = r"C:\Users\fujin\Downloads\reddit_zst\reddit\submissions"

# Output file path
output_file_path = r"C:\Users\fujin\Downloads\test\all_authors.txt"

# Set to True if you want to include duplicates, False otherwise
allow_duplicates = False

# Set to True if you want to exclude authors named '[deleted]', False otherwise
exclude_deleted_authors = True

# List of authors
authors = set()

# Function to read lines from a ZST file and extract authors
def read_zst_file(file_path):
    with open(file_path, 'rb') as file_handle:
        decompressor = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = decompressor.stream_reader(file_handle)
        chunk_size = 2**20  # 1MB chunk size (adjust as per your needs)
        buffer = b""
        file_size = os.fstat(file_handle.fileno()).st_size
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc=file_path)
        while True:
            chunk = stream_reader.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            lines = buffer.split(b"\n")
            for line in lines[:-1]:
                try:
                    obj = json.loads(line.decode())
                    author = obj['author']
                    if exclude_deleted_authors and author == '[deleted]':
                        continue
                    authors.add(author)
                except (json.JSONDecodeError, KeyError):
                    continue
            buffer = lines[-1]
            pbar.update(len(chunk))
        pbar.close()

# Iterate over all ZST files in the directory
file_count = 0
for file_name in os.listdir(zst_directory_path):
    if file_name.endswith(".zst"):
        file_path = os.path.join(zst_directory_path, file_name)
        read_zst_file(file_path)
        file_count += 1

# Write authors to the output file with progress checks
with open(output_file_path, 'w', encoding='UTF-8') as output_file:
    total_authors = len(authors)
    pbar = tqdm(authors, total=total_authors, desc="Writing to TXT")
    for author in pbar:
        output_file.write(author + '\n')
        pbar.set_postfix({"Total Authors": total_authors, "Current Author": author})
    pbar.close()

print(f"Processed {file_count} ZST files. Found {total_authors} unique authors.")
