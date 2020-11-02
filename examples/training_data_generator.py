from numbers import Number
from typing import Callable, Dict

import networkx as nx
import pandas as pd
from scipy.stats import bernoulli


class Criterion:
    """
    Stores a lambda function of the form lambda x: x <operator> c for use in defining branches in a decision tree for
    generating values from
    """

    implemented_operators = (">", "<=")

    def __init__(self, cutoff: Number, comparison_operator: str):
        self.cutoff = cutoff
        assert (
            comparison_operator in self.implemented_operators
        ), "Comparison operator was {!r} but needs to be one of {!r}".format(
            comparison_operator, self.implemented_operators
        )
        self.comparison_operator = comparison_operator

    def __str__(self):
        """
        Just show the criterion, assuming a lambda function
        """
        return "{} {}".format(self.comparison_operator, self.cutoff)

    def __repr__(self):
        return self.__str__()


def criteria_dict_to_row_values(
    criteria: Dict[str, Criterion], default_row_values: Dict[str, Number]
) -> Dict[str, Number]:
    """
    Given a dictionary with column_name -> Criteria mapping, output what to set the feature row values to.
    """
    output_row_values = default_row_values.copy()
    for column_name, criterion in criteria.items():
        if criterion.comparison_operator == "<=":
            output_row_values.update(
                {column_name: criterion.cutoff}
            )  # set feature value to exactly the cutoff if <=
        elif criterion.comparison_operator == ">":
            output_row_values.update(
                {column_name: criterion.cutoff + 1}
            )  # set feature value to cutoff + 1  if >
        else:
            raise ValueError("Unsupported {}".format(criterion.comparison_operator))

    return output_row_values


def tree_to_leaf_row_values(
    tree: nx.DiGraph, row_value_fn: Callable, default_row_values: Dict[str, Number]
) -> Dict[str, Dict[str, Number]]:
    """
    Return, for list of leaf nodes, a dictionary mapping column name to value to generate. Note: not implemented for
    recurring instances of the same feature along a single path to a leaf.
    """
    leaf_row_values = {}
    # Get root name
    root_name = [n for (n, d) in tree.in_degree() if d == 0][0]
    # Get criteria for each leaf
    for node in tree:
        if tree.out_degree(node) == 0:  # leaf node
            criteria = {}
            for path in next(
                nx.all_simple_edge_paths(tree, root_name, node)
            ):  # only one path, so calling next() once gets it (math.stackexchange.com/a/1523566/440173)
                criteria[path[0]] = tree.edges[path]["criterion"]

            # Translate criteria to row value dict
            leaf_row_values[node] = row_value_fn(criteria, default_row_values)

    return leaf_row_values


def generate_rows_per_leaf(
    tree: nx.DiGraph, default_row_values, outcome_name: str, n_per_leaf=100, seed=42,
) -> pd.DataFrame:
    """
    For a given networkx tree constructed according to the API (with probabilities of `outcome_name` at each leaf),
    sample row values from each leaf.
    """

    row_values_per_leaf = tree_to_leaf_row_values(
        tree=tree,
        row_value_fn=criteria_dict_to_row_values,
        default_row_values=default_row_values,
    )

    dfs = []
    for leaf_name, leaf_row_values in row_values_per_leaf.items():
        leaf_df = pd.DataFrame(
            columns=list(leaf_row_values.keys()) + ["{}_label".format(outcome_name)],
            index=range(n_per_leaf),
        )
        leaf_df.loc[range(n_per_leaf), leaf_row_values.keys()] = list(
            leaf_row_values.values()
        )

        # Sample based on p_delinquency
        leaf_df.loc[range(n_per_leaf), "{}_label".format(outcome_name)] = bernoulli.rvs(
            p=tree.nodes[leaf_name]["p_{}".format(outcome_name)],
            size=n_per_leaf,
            random_state=seed,
        )
        dfs.append(leaf_df)

    return pd.concat(dfs, ignore_index=True)
