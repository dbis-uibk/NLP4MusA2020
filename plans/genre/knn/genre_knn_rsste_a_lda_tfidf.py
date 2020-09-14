"""Plan for a knn model, rhymes, statist., statist. time, explicit, audio."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nlp4musa2020.dataloaders.alf200k import ALF200KLoader
from nlp4musa2020.dataloaders.vectorizer import lda
from nlp4musa2020.dataloaders.vectorizer import tfidf
import nlp4musa2020.evaluators as evaluators

dataloader = ALF200KLoader(
    path='data/processed/dataset-lfm-genres.pickle',
    load_feature_groups=[
        'rhymes',
        'statistical',
        'statistical_time',
        'explicitness',
        'audio',
    ],
    text_vectorizers=lda() + tfidf(),
    target=[
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
    ],
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree')),
])

evaluator = GridEvaluator(
    parameters={
        'model__n_neighbors': [3, 4, 5, 10],
        'model__weights': ['distance'],
        'model__p': [1, 2],
    },
    grid_parameters=evaluators.grid_parameters_genres(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
