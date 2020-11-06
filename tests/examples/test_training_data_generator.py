import unittest

import networkx as nx
import pandas as pd

from examples.training_data_generator import (
    criteria_dict_to_row_values,
    Criterion,
    tree_to_leaf_row_values,
    generate_rows_per_leaf,
)


class TestCriterion(unittest.TestCase):
    def test_construct(self):
        with self.assertRaises(AssertionError):
            Criterion(30, "~")

        self.assertEqual("<= 10", str(Criterion(10, "<=")))


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.criteria = [
            Criterion(10, "<="),
            Criterion(1, ">"),
        ]
        self.criteria_dict = dict(zip(("foo", "bar"), self.criteria))
        self.default_row_values = {"foo": 0, "baz": 2}

        # foo -(<= 10)-> bar --(> 1) > baz
        self.three_node_tree = nx.DiGraph()
        self.three_node_tree.add_edge(
            "foo", "bar", criterion=self.criteria[0],
        )
        self.three_node_tree.add_edge("bar", "baz", criterion=self.criteria[1])

        self.outcome_name = "that_happened"

        self.three_node_tree.nodes["baz"]["p_{}".format(self.outcome_name)] = 0.1

    def test_criteria_dict_to_row_values(self):
        row_values = criteria_dict_to_row_values(
            self.criteria_dict, default_row_values=self.default_row_values
        )
        self.assertEqual({"foo": 10, "bar": 2, "baz": 2}, row_values)

    def test_tree_to_leaf_row_values(self):
        row_values_per_leaf = tree_to_leaf_row_values(
            tree=self.three_node_tree,
            row_value_fn=criteria_dict_to_row_values,
            default_row_values=self.default_row_values,
        )
        self.assertEqual({"baz": {"foo": 10, "baz": 2, "bar": 2}}, row_values_per_leaf)

    def test_generate_rows_per_leaf(self):
        df = generate_rows_per_leaf(
            tree=self.three_node_tree,
            default_row_values=self.default_row_values,
            outcome_name=self.outcome_name,
            n_per_leaf=2,
            seed=42,
        )

        #    foo  baz  bar  that_happened_label
        # 0   10    2    2                    0
        # 1   10    2    2                    1
        expected = pd.DataFrame.from_dict(
            {
                "foo": {0: 10, 1: 10},
                "baz": {0: 2, 1: 2},
                "bar": {0: 2, 1: 2},
                "that_happened_label": {0: 0, 1: 1},
            }
        )
        pd.testing.assert_frame_equal(expected, df, check_dtype=False)
