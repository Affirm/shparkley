from __future__ import absolute_import
from __future__ import division
from typing import TYPE_CHECKING
import numpy as np
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from pyspark.sql import functions as fn
from itertools import permutations, cycle, chain
from future.builtins import zip, range

if TYPE_CHECKING:
    from pyspark.sql import Row, DataFrame
    from typing import List, Iterable, Tuple, Dict, Set, Generator, Optional, Any


@add_metaclass(ABCMeta)
class ShparkleyModel(object):
    """
    Abstract class for computing Shapley values.
    """
    def __init__(self, model):
        # type: (Any) -> None
        self._model = model

    @abstractmethod
    def get_required_features(self):
        # type: () -> Set[str]
        """
        Returns the set of feature names
        :return: Set of feature names
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature_matrix):
        # type: (List[Dict[str, Any]]) -> List[float]
        """
        Run the machine learning model on a feature matrix and return the predictions for each row.
        :param feature_matrix: Row of feature vectors. Each entry is a dictionary mapping
        from the feature name to feature value
        :return: Model predictions for all feature vectors
        """
        raise NotImplementedError


def compute_shapley_score(partition_index, rand_rows, row_to_investigate, model, weight_col_name=None):
    # type: (int, Iterable[Row], Row, ShparkleyModel, Optional[str]) -> Generator[Tuple[str, float, float], None, None]
    """
    Computes the shapley marginal contribution for each feature in the feature vector over all
    samples in the partition.
    The algorithm is based on a monte-carlo approximation:
    https://christophm.github.io/interpretable-ml-book/shapley.html#fn42
    :param partition_index: Index of spark partition which will serve as a seed to numpy
    :param rand_rows: Sampled rows of the dataset in the partition
    :param row_to_investigate: Feature vector for which we need to compute shapley scores
    :param model: ShparkleyModel object which implements the predict function.
    :param weight_col_name: column name with row weights to use when sampling the training set
    :return: Generator of tuple of feature and shapley marginal contribution
    """
    required_features = list(model.get_required_features())
    random_feature_permutation = np.random.RandomState(partition_index).permutation(required_features)
    # We cycle through permutations in cases where the number of samples is more than
    # the number of features
    permutation_iter = cycle(permutations(random_feature_permutation))
    feature_vector_rows = []
    rand_row_weights = []
    for rand_row in rand_rows:  # take sample z from training set
        rand_row_weights.append(rand_row[weight_col_name] if weight_col_name is not None else 1)
        feature_permutation = next(permutation_iter)  # choose permutation o
        # gather: {z_1, ..., z_p}
        feat_vec_without_feature = {feat_name: rand_row[feat_name] for feat_name in required_features}
        feat_vec_with_feature = feat_vec_without_feature.copy()
        for feat_name in feature_permutation:  # for random feature k.
            # x_+k = {x_1, ..., x_k, .. z_p}
            feat_vec_with_feature[feat_name] = row_to_investigate[feat_name]
            # x_-k = {x_1, ..., x_{k-1}, z_k, ..., z_p}
            # store (x_+k, x_-k)
            feature_vector_rows.append(feat_vec_with_feature.copy())
            feature_vector_rows.append(feat_vec_without_feature.copy())
            # (x_-k = x_+k)
            feat_vec_without_feature[feat_name] = row_to_investigate[feat_name]

    if len(feature_vector_rows) == 0:
        return
    preds = model.predict(feature_vector_rows)
    feature_iterator = chain.from_iterable(cycle(permutations(random_feature_permutation)))
    for pred_index, feature in zip((range(0, len(preds), 2)), feature_iterator):
        marginal_contribution = preds[pred_index] - preds[pred_index + 1]

        #  There is one weight added per random row visited.
        #  For each random row visit, we generate 2 predictions for each required feature.
        #  Therefore, to get index into rand_row_weights, we need to divide
        #  prediction index by 2 * number of features, and take the floor of this.
        weight = rand_row_weights[pred_index // (len(required_features) * 2)]

        yield (str(feature), float(marginal_contribution), float(weight))


def compute_shapley_for_sample(df, model, row_to_investigate, weight_col_name=None):
    # type: (DataFrame, ShparkleyModel, Row, Optional[str]) -> Dict[str, float]
    """
    Compute shapley values for all features in a given feature vector of interest.

    :param df: Training dataset
    :param model: ShparkleyModel object which implements the predict function.
    :param row_to_investigate: Feature vector for which we need to compute shapley scores
    :param weight_col_name: column name with row weights to use when sampling the training set
    :return: Dictionary of feature mapping to its corresponding shapley value.
    """
    shapley_df = (
        df.rdd.mapPartitionsWithIndex(
            lambda idx, rows: compute_shapley_score(idx, rows, row_to_investigate, model, weight_col_name),
            preservesPartitioning=True,
        )
    ).toDF(["feature", "marginal_contribution", "weight"])

    return dict(
        shapley_df.groupBy("feature")
                  .agg((
                                 fn.sum(shapley_df.marginal_contribution * shapley_df.weight) /
                                 fn.sum(shapley_df.weight)
                    ).alias("shapley_value"))
                  .collect()
    )
