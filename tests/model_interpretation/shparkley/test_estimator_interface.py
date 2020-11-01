from __future__ import division
from affirm.model_interpretation.shparkley.estimator_interface import (
    OrderedSet,
    ShparkleyModel,
)
import unittest
from random import shuffle


class TestOrderedSet(unittest.TestCase):
    def setUp(self):
        self.foo_list = list(range(10))
        shuffle(self.foo_list)

    def test_roundtrip(self):
        self.assertEqual(self.foo_list, list(OrderedSet(self.foo_list)))

        self.assertEqual(
            OrderedSet(self.foo_list), OrderedSet(list(OrderedSet(self.foo_list)))
        )

    def test_ignore_duplicates(self):
        added_orderd_set = OrderedSet(self.foo_list)
        added_orderd_set.add(1)  # does not reorder, just ignores as already present
        self.assertEqual(OrderedSet(self.foo_list), added_orderd_set)

    def test_update(self):
        start = OrderedSet(self.foo_list)
        start.update([10, 11])
        self.assertEqual(start, OrderedSet(self.foo_list + [10, 11]))

    def test_discard(self):
        start = OrderedSet(self.foo_list)
        start.discard(0)
        self.assertEqual(OrderedSet(self.foo_list[1:]), start)


class TestShparkleyModel(unittest.TestCase):
    def test_api(self):
        class UnorderedRequiredFeaturesShparkleyModel(ShparkleyModel):
            def _get_required_features(self):
                return {"f1"}

            def predict(self, feature_matrix):
                pass

        class OrderedRequiredFeaturesShaprkleyModel(ShparkleyModel):
            def _get_required_features(self):
                return OrderedSet(["f1"])

            def predict(self, feature_matrix):
                pass

        with self.assertRaises(AssertionError):
            UnorderedRequiredFeaturesShparkleyModel(model=None).get_required_features()

        OrderedRequiredFeaturesShaprkleyModel(model=None).get_required_features()
