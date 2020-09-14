import pickle
import argparse

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# We use the 50 most common tags in our dataset, and from those take only
# tags that represent genres.
# Additionally, we "merge" subgenres with the parent genre based on a simple
# lexicographic mapping. For example, "alternative rock" is "rock" for us.
# We also remove the "acoustic" genre, since ... Well, it does not make any sense
# for our setting?!
genres = {
    "pop": "pop",
    "rap": "rap",
    "rock": "rock",
    "hip hop": "hip hop",
    "hip-hop": "hip hop",
    "indie": "indie",
    "alternative": "alternative",
    "alternative rock": "rock",
    "classic rock": "rock",
    "indie rock": "rock",
    "soul": "soul",
    "electronic": "electronic",
    "hard rock": "rock",
    "metal": "metal",
    "country": "country",
    "rnb": "rnb",
    "punk": "punk",
    "dance": "dance",
    "indie pop": "pop",
    "jazz": "jazz",
    "blues": "blues",
    "punk rock": "rock",
    "heavy metal": "metal",
    "funk": "funk",
}


def extract_genres(df):
    # Convert all tags to lower case.
    df["tags"] = df["tags"].apply(
        lambda song_tags: [t.lower() for t in song_tags])

    # Reduce tag list according to our tag mapping dictionary.
    df["tags"] = df["tags"].apply(
        lambda song_tags: [genres[tag] for tag in song_tags if tag in genres])

    # Convert tag list to None if a song does not have any valid genres. This makes dropping those
    # songs easier.
    df["tags"] = df["tags"].apply(lambda song_tags: song_tags
                                  if song_tags != [] else None)

    # Drop rows that do not have a valid tag.
    df = df.dropna()

    # Binarize.
    mlb = MultiLabelBinarizer()
    indicator_vector = mlb.fit_transform(df["tags"])
    df_indicator = pd.DataFrame(indicator_vector,
                                columns=mlb.classes_,
                                index=df.index)

    df = pd.concat([df, df_indicator], axis=1)

    # Done.
    return df


def main():
    # Load input file.
    df = pickle.load(open(args.input, "rb"))

    # Extract the genre tags from all LFM tags.
    df = extract_genres(df)

    # Save result.
    pickle.dump(df, open(args.output, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        dest="input",
                        required=True,
                        help="The path to the pickled LFM dataset with tags.")
    parser.add_argument("--output",
                        dest="output",
                        required=True,
                        help="The path to the result file.")
    args = parser.parse_args()

    main()
