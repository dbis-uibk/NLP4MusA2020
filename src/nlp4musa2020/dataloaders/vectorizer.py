"""Commonly used predefined vectorizers."""
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf(max_features=2_000):
    """List of tf-idf vectorizers.

    Args:
        max_features: number of features to consider.
    """
    return [
        tfidf_word(max_features=max_features),
        tfidf_char(max_features=max_features),
    ]


def tfidf_word(max_features=2_000):
    """Word tf-idf vectorizer.

    Args:
        max_features: number of features to consider.
    """
    return TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer='word',
        max_features=max_features,
    )


def tfidf_char(max_features=2_000):
    """Char tf-idf vectorizer.

    Args:
        max_features: number of features to consider.
    """
    return TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer='char',
        max_features=max_features,
    )


def ngram(max_features=2_000):
    """List of ngram vectorizers.

    Args:
        max_features: number of features to consider.
    """
    return [
        ngram_word(max_features=max_features),
        ngram_char(max_features=max_features),
    ]


def ngram_word(max_features=2_000):
    """Word count vectorizer.

    Args:
        max_features: number of features to consider.
    """
    return CountVectorizer(
        ngram_range=(1, 3),
        analyzer='word',
        max_features=max_features,
    )


def ngram_char(max_features=2_000):
    """Char count vectorizer.

    Args:
        max_features: number of features to consider.
    """
    return CountVectorizer(
        ngram_range=(1, 3),
        analyzer='char',
        max_features=max_features,
    )


def lda():
    """LDA vectorizer."""
    return [LDAVectorizer()]


class LDAVectorizer:
    """LDA vectorizer impementation."""

    def __init__(self):
        """Initializers the vectorizer."""
        pass

    def fit_transform(self, X):  # noqa: N803
        """Computes the features for X."""
        # Get word counts per document.
        cv = CountVectorizer()
        counts = cv.fit_transform(X)

        # Fit LDA.
        self.lda = LatentDirichletAllocation(n_components=25)
        return self.lda.fit_transform(counts)
