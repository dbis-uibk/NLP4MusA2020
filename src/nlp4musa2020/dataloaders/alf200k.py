"""Dataloader for the ALF 200k dataset."""
import pickle

from dbispipeline.base import Loader
import numpy as np


def pearson_correlated_20():
    """Returns a list of features which a pearson correlation >0.2.

    The correlation is cumputed between the feature and popularity.
    """
    return [
        'token_count',
        'unique_token_ratio',
        'repeat_word_ratio',
        'line_count',
        'unique_line_count',
        'hapax_legomenon_ratio',
        'words_per_minute',
        'chars_per_minute',
        'lines_per_minute',
        'explicit',
    ]


def genre_target_labels():
    """Returns a list of genre target labels used for classification."""
    return [
        'alternative',
        'blues',
        'country',
        'dance',
        'electronic',
        'funk',
        'hip hop',
        'indie',
        'jazz',
        'metal',
        'pop',
        'punk',
        'rap',
        'rnb',
        'rock',
        'soul',
    ]


class ALF200KLoader(Loader):
    """Loads the ALF200K dataset from a pickled dataframe."""

    feature_groups = {
        'rhymes': [
            'rhymes_per_line',
            'rhymes_per_syllable',
            'rhyme_density',
            'end_pairs_per_line',
            'singles_per_rhyme',
            'doubles_per_rhyme',
            'triples_per_rhyme',
            'quads_per_rhyme',
            'longs_per_rhyme',
            'perfect_rhymes',
            'line_internals_per_line',
            'links_per_line',
            'bridges_per_line',
            'compounds_per_line',
            'chaining_per_line',
        ],
        'statistical': [
            'token_count',
            'unique_token_ratio',
            'unique_bigram_ratio',
            'unique_trigram_ratio',
            'average_token_length',
            'unique_tokens_per_line',
            'average_tokens_per_line',
            'repeat_word_ratio',
            'line_count',
            'unique_line_count',
            'blank_line_count',
            'blank_line_ratio',
            'repeat_line_ratio',
            'digits',
            'exclamation_marks',
            'question_marks',
            'colons',
            'semicolons',
            'quotes',
            'commas',
            'dots',
            'hyphens',
            'stopwords_ratio',
            'stopwords_per_line',
            'hapax_legomenon_ratio',
            'dis_legomenon_ratio',
            'tris_legomenon_ratio',
            'syllables_per_line',
            'syllables_per_word',
            'syllable_variation',
            'novel_word_proportion',
        ],
        'statistical_time': [
            'words_per_minute',
            'chars_per_minute',
            'lines_per_minute',
        ],
        'explicitness': ['explicit'],
        'audio': [
            'tempo',
            'energy',
            'liveness',
            'speechiness',
            'acousticness',
            'danceability',
            'loudness',
            'valence',
            'instrumentalness',
            'duration',
        ],
    }

    def __init__(self,
                 path,
                 load_feature_groups=None,
                 text_vectorizers=None,
                 target='popularity',
                 features=None,
                 drop_duplicates=True):
        """Intitializes the dataloader object.

        Parameters:
            path (str): The path to the pickled dataframe containing the
                ALF200k dataset.
            feature_groups (list): The list of feature groups to load. If None
                (default), load all features.
            text_vectorizers (list): List of feature vectorizers to run on the
                lyrics texts.
            features (list): if set, the feature_groups are ignored and the
                loader selects those features.
            drop_duplicates (bool): drops duplicates based on title and artist.
        """
        self.path = path
        self.load_feature_groups = load_feature_groups
        self.text_vectorizers = text_vectorizers
        self.target = target
        self.drop_duplicates = drop_duplicates

        if features is not None:
            self.features = features
        else:
            self.features = []
            for fg in load_feature_groups:
                self.features.extend(self.feature_groups[fg])

    def load(self):
        """Load the dataset as a dataframe."""
        # Load the dataframe from the pickle file.
        df = pickle.load(open(self.path, 'rb'))

        if self.drop_duplicates is True:
            df = df.drop_duplicates(['name', 'artist_name'])

        # Extract the popularity as the target label.
        y = df[self.target].to_numpy()

        # Load the selected features.
        X = df[self.features]  # noqa: N806

        if self.text_vectorizers is not None:
            data = [X.to_numpy().astype('float64')]
            for vectorizer in self.text_vectorizers:
                vectorized = vectorizer.fit_transform(df['text'])
                try:
                    vectorized = vectorized.todense().astype('float64')
                except AttributeError:
                    pass
                data.append(vectorized)
            X = np.hstack(data)  # noqa: N806

        # Done.
        return X, y

    @property
    def configuration(self):
        """Returns a dict-like representation of the configuration."""
        return {
            'name': 'ALF200KLoader',
            'path': self.path,
            'load_feature_groups': self.load_feature_groups,
            'features': self.features,
            'text_vectorizers': str(self.text_vectorizers),
            'target': self.target,
            'drop_duplicates': self.drop_duplicates,
        }
