import json
import pickle
import argparse
import json
import urllib.request
import time
from urllib.parse import quote
from urllib.error import HTTPError
from os.path import join

import pandas as pd
import fuzzymatcher
from pandas import json_normalize


def load_lyrics_json(filename):
    with open(join(args.lyrics_dir, filename), "r") as f:
        return json_normalize(json.load(f))


def merge_lyrics_to_df():
    # Load the lyrics JSON files into dataframes.
    df_tracks = load_lyrics_json("tracks.json")
    df_lyrics = load_lyrics_json("lyrics.json")
    df_opinions = load_lyrics_json("opinions.json")
    df_rhymes = load_lyrics_json("rhymes.json")
    df_textstyles = load_lyrics_json("textstyles.json")
    df_audios = load_lyrics_json("audios.json")

    # Merge the dataframes.
    df = df_tracks.merge(df_lyrics,
                         left_on="_id.$oid",
                         right_on="track_id.$oid",
                         suffixes=("_tracks", "_lyrics"))
    df = df.merge(df_audios,
                  left_on="_id.$oid_tracks",
                  right_on="track_id.$oid",
                  suffixes=("", "_audios"))
    df = df.merge(df_opinions,
                  left_on="_id.$oid_lyrics",
                  right_on="lyric_id.$oid",
                  suffixes=("", "_opinions"))
    df = df.merge(df_textstyles,
                  left_on="_id.$oid_lyrics",
                  right_on="lyric_id.$oid",
                  suffixes=("", "_textstyles"))
    df = df.merge(df_rhymes,
                  left_on="_id.$oid_lyrics",
                  right_on="lyric_id.$oid",
                  suffixes=("", "_rhymes"))

    # Extract the artist name.
    artists = df["artists"].tolist()
    artist_names = [a[0]["name"] for a in artists]
    df["artist_name"] = artist_names

    # Remove unwanted columns.
    del df["available_markets"]
    del df["type"]
    del df["artists"]
    del df["preview_url"]
    del df["track_number"]
    del df["href"]
    del df["id"]
    del df["_id.$oid_tracks"]
    del df["album.images"]
    del df["album.name"]
    del df["album.available_markets"]
    del df["album.album_type"]
    del df["album.href"]
    del df["album.id"]
    del df["album.type"]
    del df["album.external_urls.spotify"]
    del df["album.uri"]
    del df["disc_number"]
    del df["uri"]
    del df["external_ids.isrc"]
    del df["external_urls.spotify"]
    del df["url"]
    del df["source"]
    del df["_id.$oid_lyrics"]
    del df["track_id.$oid"]
    del df["_id.$oid"]
    del df["lyric_id.$oid"]
    del df["_id.$oid_textstyles"]
    del df["lyric_id.$oid_textstyles"]
    del df["_id.$oid_rhymes"]
    del df["lyric_id.$oid_rhymes"]
    del df["track_id.$oid_audios"]
    del df["_id.$oid_opinions"]

    # Remove columns with NaN values.
    columns = list(df.columns)
    for column in columns:
        if df[column].isnull().values.any():
            del df[column]

    # Get only about 50k songs.
    df = df.sample(frac=0.25, replace=False, random_state=42)

    # Done.
    return df


def add_lfm_data(df):
    print("Adding LFM data ...")

    # Add placeholder value.
    df["playcount"] = -1
    df["tags"] = None

    # Iterate the rows in the dataframe and try to get the playcount for every row.
    for idx, row in df.iterrows():
        # Get API response.
        artist_name = row["artist_name"]
        track_name = row["name"]

        try:
            api_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key=f7de90bc23fa4720f140f30ca9c139cd&track={quote(track_name)}&artist={quote(artist_name)}&format=json"
            api_resp = urllib.request.urlopen(api_url)
            resp_string = api_resp.read().decode("utf8")
        except:
            print(f"    !!! Error for song {artist_name} - {track_name}")
            continue

        time.sleep(0.25)

        # Skip if an error occured.
        if '"error"' in resp_string:
            continue

        # Load as JSON.
        lfm_data = json.loads(resp_string)

        # Skip if no tags.
        if "toptags" not in lfm_data["track"]:
            continue

        # Add data to dataframe.
        playcount = lfm_data["track"]["playcount"]
        df.at[idx, "playcount"] = playcount

        tags = [x["name"] for x in lfm_data["track"]["toptags"]["tag"]]
        df.at[idx, "tags"] = tags

        print(
            f"    Found data for {artist_name} - {track_name}, playcount {playcount}, tags {tags} ..."
        )

    # Remove all rows that don't have a proper playacount.
    # These are all rows for which LFM did not return any data.
    df = df[df["playcount"] != -1]

    return df


def main():
    # Load data.
    df_lyrics = merge_lyrics_to_df()

    # Add Last.fm data.
    df_lyrics = add_lfm_data(df_lyrics)

    # Save result.
    pickle.dump(df_lyrics,
                open(join(args.output_dir, "dataset-lfm-genres.pickle"), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lyrics-dir",
        dest="lyrics_dir",
        required=True,
        help="The directory in which the lyrics JSON files are located.")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="The directory in which to store the rinal result.")
    args = parser.parse_args()

    main()
