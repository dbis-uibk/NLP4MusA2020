"""Config for a linear regression model evaluated on a diabetes dataset."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nlp4musa2020.dataloaders.alf200k import ALF200KLoader
from nlp4musa2020.dataloaders.alf200k import genre_target_labels
from nlp4musa2020.dataloaders.vectorizer import lda
from nlp4musa2020.dataloaders.vectorizer import tfidf
import nlp4musa2020.evaluators as evaluators
from nlp4musa2020.models.simplenn_genre import SimpleGenreNN

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
    target=genre_target_labels(),
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SimpleGenreNN(epochs=50)),
])

evaluator = GridEvaluator(
    parameters={
        'model__dense_sizes': [
            (32, 32),
            (64, 64),
        ],
        'model__dropout_rate': [0.1],
    },
    grid_parameters=evaluators.grid_parameters_genres(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
