"""Config for a linear regression model evaluated on a diabetes dataset."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from nlp4musa2020.dataloaders.alf200k import ALF200KLoader, genre_target_labels
from nlp4musa2020.dataloaders.vectorizer import tfidf
import nlp4musa2020.evaluators as evaluators
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier

dataloader = ALF200KLoader(
    'data/processed/dataset-lfm-genres.pickle',
    load_feature_groups=[],
    text_vectorizers=tfidf(),
    target=genre_target_labels()
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MultiOutputClassifier(LinearSVC())),
])

evaluator = GridEvaluator(
    parameters={
        'model__estimator__C': [
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
        ],
        'model__estimator__loss': ['epsilon_insensitive'],
    },
    grid_parameters=evaluators.grid_parameters_genres(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
