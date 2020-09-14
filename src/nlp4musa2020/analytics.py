"""Module providing common functions used for analytics."""
import os.path

from dbispipeline.analytics import extract_gridsearch_parameters
from dbispipeline.db import DB
import matplotlib.pyplot as plt
import pandas as pd


def get_results(project_name, filter_git_dirty=True):
    """Returns the results stored in the databes as a pandas dataframe.

    Args:
        project_name: The project name to fetch results.
        filter_git_dirty: defines if dirty commits are filterd.
    """
    results = pd.read_sql_table(table_name='results', con=DB.engine)

    if filter_git_dirty:
        results = results[results['git_is_dirty'] == False]  # noqa: E712

    return results[results['project_name'] == project_name]


def extract_best_result(result, score):
    """Extracts the max value result for a given score column.

    Args:
        result: dataframe to extract results from.
        score: the column used to select the max value.
    """
    result = result[result[score] >= result[score].max()]
    return result


def get_best_results(score_name, score_prefix='mean_test_', max_value=None):
    """Returns the best results for a given score.

    Args:
        score_name: the name of the score that is prefixed with score_prefix.
            The result is the column used to extract the results.
        score_prefix: the prefix of the score name.
        max_value: If not None, include only results that have scores lower
            than this value.
    """
    data = get_results(project_name='nlp4musa2020')

    score = score_prefix + score_name

    result = pd.DataFrame()
    for _, group in data.groupby(['sourcefile']):
        outcome = None
        try:
            outcome = extract_gridsearch_parameters(group, score_name=score)
        except KeyError:
            continue
        # FIXME: Remove after updating the dbispipeline
        outcome[score] = outcome['score']
        if '_neg_' in score:
            outcome[score] *= -1

        result = result.append(extract_best_result(outcome, score=score))

    if max_value is not None:
        result = result[result[score] < max_value]

    if len(result) < 1:
        raise Exception('No results found.')

    return result


def plot_best_results(score_name,
                      score_prefix='mean_test_',
                      max_value=None,
                      result_path=None,
                      file_ext='pdf'):
    """Plots the to results for a given metric.

    Args:
        score_name: the name of the score that is prefixed with score_prefix.
            The result is the column used to extract the results.
        score_prefix: the prefix of the score name.
        max_value: If not None, include only results that have scores lower
            than this value.
        result_path: the path used to store result files.
        file_ext: the file extension used for the plots.
    """
    result = get_best_results(score_name, score_prefix, max_value)

    score = score_prefix + score_name
    result[['sourcefile', score]].plot.bar(
        x='sourcefile',
        title=score_name,
        grid=True,
        figsize=(30, 10),
    )

    file_name = 'best_results_' + score_name + '.' + file_ext
    if result_path is not None:
        file_name = os.path.join(result_path, file_name)

    plt.savefig(file_name)
