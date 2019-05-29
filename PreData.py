import csv
import pandas as pd

def data_cleanup():
    # Cleaning up the data
    with open('lyrics.csv', 'r',encoding="utf8") as inp, open('lyrics_out.csv', 'w',encoding="utf8",newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if (row[4] == "genre" or row[4] == "Pop" or row[4] == "Rock" or row[4] == "Country" or row[4] == "Hip-Hop" or row[4] == "Jazz") and row[5] != "":
                writer.writerow(row)

def data_filter():
    data_cleanup()
    data = pd.read_csv('lyrics_out.csv',usecols=['index', 'genre', 'lyrics'],index_col='index')
    data = data.groupby('genre' ).head(7000)
    data.to_csv('lyrics_out.csv')
    df = pd.read_csv('lyrics_out.csv',usecols=['genre', 'lyrics'])
    df['word_count'] = df['lyrics'].str.split().str.len()
    df["lyrics"] = df['lyrics'].str.lower()
    df['lyrics'] = df['lyrics'].str.strip('[]')
    df['lyrics'] = df['lyrics'].str.strip('()')
    df["lyrics"] = df['lyrics'].str.replace('[^\w\s]', '')
    df["lyrics"] = df['lyrics'].str.replace('chorus', '')
    df["lyrics"] = df['lyrics'].str.replace(':', '')
    df["lyrics"] = df['lyrics'].str.replace(',', '')
    df["lyrics"] = df['lyrics'].str.replace('verse', '')
    df["lyrics"] = df['lyrics'].str.replace('x1', '')
    df["lyrics"] = df['lyrics'].str.replace('x2', '')
    df["lyrics"] = df['lyrics'].str.replace('x3', '')
    df["lyrics"] = df['lyrics'].str.replace('x4', '')
    df["lyrics"] = df['lyrics'].str.replace('x5', '')
    df["lyrics"] = df['lyrics'].str.replace('x6', '')
    df["lyrics"] = df['lyrics'].str.replace('x7', '')
    df["lyrics"] = df['lyrics'].str.replace('x8', '')
    df["lyrics"] = df['lyrics'].str.replace('x9', '')
    df = df[df['word_count'] > 100]
    del df['word_count']
    df["lyrics"] = df['lyrics'].str.lower().str.replace('[^\w\s]', '')
    df = df.groupby('genre', ).head(5000)
    df.to_csv('lyrics_final.csv', index=False)

if __name__ == '__main__':
    data_filter()

