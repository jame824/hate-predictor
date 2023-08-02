import random

def get_names(input, output, num_names):
    with open(input, 'r') as f:
        names = f.read().splitlines()
    selected_names = random.sample(names, num_names)

    with open(output, 'w') as f:
        f.write(','.join(selected_names))

if __name__ == '__main__':
    author_file = r"C:\Users\fujin\Downloads\test\all_authors.txt"
    output_path = r"C:\Users\fujin\Downloads\test\20k_authors.txt"
    amount = 20000
    get_names(author_file, output_path, amount)