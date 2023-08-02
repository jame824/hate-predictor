import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("../isi/whiterights_commentsfilter60.csv", header=None)
    df.columns = ["uknwn", "date", "uid", "link", "text"]
    df["date"] = pd.to_datetime(df["date"])

    df2 = df.groupby("uid")
    first_list = []
    for uid in df["uid"].drop_duplicates():
        first = df2.get_group(uid)["date"].min()
        first_list.append(first)

    first_counts = pd.Series(first_list).value_counts().sort_index()  # Count instances per day

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(first_counts.index, first_counts.values, marker='o', linestyle='-', color='b')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('First Comment Counts per Day')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
