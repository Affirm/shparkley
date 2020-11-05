from abc import ABCMeta, abstractmethod

from six import add_metaclass
from typing import Any, Dict, List

from collections import OrderedDict
from collections.abc import MutableSet


@add_metaclass(ABCMeta)
class ShparkleyModel(object):
    """
    Abstract class for computing Shapley values.
    """

    def __init__(self, model):
        # type: (Any) -> None
        self._model = model

    def get_required_features(self):
        # type: () -> OrderedSet[str]
        """
        Returns an ordered set of feature names (ordered so that when recreating a
        feature matrix from a list of per-row dicts, the columns will be in the expected order)
        :return: OrderedSet of feature names
        """
        features = self._get_required_features()
        assert isinstance(
            features, OrderedSet
        ), "required_features must be an OrderedSet (was {!r} instead)".format(
            type(features)
        )
        return features

    @abstractmethod
    def _get_required_features(self):
        # type: () -> OrderedSet[str]
        """
        Returns an ordered set of feature names (ordered so that when recreating a
        feature matrix from a list of per-row dicts, the columns will be in the expected order).
        :return: OrderedSet of feature names
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature_matrix):
        # type: (List[OrderedDict[str, Any]]) -> List[float]
        """
        Run the machine learning model on a feature matrix and return the predictions for each row. Take care to
        construct the feature matrix with the same column ordering that the model was fit on.
        :param feature_matrix: Row of feature vectors. Each entry is a dictionary mapping from the feature name to
        feature value
        :return: Model predictions for all feature vectors
        """
        raise NotImplementedError


class OrderedSet(OrderedDict, MutableSet):
    """
    Since use pyspark.sql.Row.asDict(), duplicate column names are not supported. However, column order must be
    maintained for models that do not check column names explicitly during prediction.
    """

    def __init__(self, *args, **kwargs):
        super(OrderedSet, self).__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem):
        self[elem] = None

    def discard(self, elem):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return "OrderedSet([%s])" % (", ".join(map(repr, self.keys())))

    def __str__(self):
        return "{%s}" % (", ".join(map(repr, self.keys())))
