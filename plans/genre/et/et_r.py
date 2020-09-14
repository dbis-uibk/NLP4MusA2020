"""Plan for a random forest classifier model."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nlp4musa2020.dataloaders.alf200k import ALF200KLoader
from nlp4musa2020.dataloaders.alf200k import genre_target_labels
import nlp4musa2020.evaluators as evaluators

dataloader = ALF200KLoader(
    path='data/processed/dataset-lfm-genres.pickle',
    load_feature_groups=[
        'rhymes',
    ],
    text_vectorizers=None,
    target=genre_target_labels(),
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ExtraTreesClassifier(n_jobs=-1)),
])

evaluator = GridEvaluator(
    parameters={
        'model__n_estimators': [10, 100, 300],
    },
    grid_parameters=evaluators.grid_parameters_genres(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
