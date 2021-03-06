"""Config for a linear regression model evaluated on a diabetes dataset."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

from nlp4musa2020.dataloaders.alf200k import ALF200KLoader, genre_target_labels
import nlp4musa2020.evaluators as evaluators
from nlp4musa2020.models.simplenn_genre import SimpleGenreNN

dataloader = ALF200KLoader(
    'data/processed/dataset-lfm-genres.pickle',
    load_feature_groups=[
        'statistical',
    ],
    text_vectorizers=None,
    target=genre_target_labels()
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', DummyClassifier(strategy="stratified")),
])

evaluator = GridEvaluator(
    parameters={
        "model__random_state": [42],
    },
    grid_parameters=evaluators.grid_parameters_genres(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
