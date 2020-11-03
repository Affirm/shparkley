Shparkley: Scaling Shapley Values with Spark
=============================================

.. inclusion-marker-start-do-not-remove

.. contents::

Shparkley is a PySpark implementation of
`Shapley values <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf>`_
which uses a `monte-carlo approximation <https://link.springer.com/article/10.1007/s10115-013-0679-x>`_ algorithm.

Given a dataset and machine learning model, Shparkley can compute Shapley values for all features for a feature vector.
Shparkley also handles training weights and is model-agnostic.

Installation
------------

``pip install shparkley``

Requirements
------------
You must have Apache Spark installed on your machine/cluster.


Example Usage
--------------

.. code-block:: python

    from typing import List

    from sklearn.base import ClassifierMixin

    from affirm.model_interpretation.shparkley.estimator_interface import OrderedSet, ShparkleyModel
    from affirm.model_interpretation.shparkley.spark_shapley import compute_shapley_for_sample


    class MyShparkleyModel(ShparkleyModel):
        """
        You need to wrap your model with this interface (by subclassing ShparkleyModel)
        """
        def __init__(self, model: ClassifierMixin, required_features: OrderedSet):
            self._model = model
            self._required_features = required_features

        def predict(self, feature_matrix: List[OrderedDict]) -> List[float]:
            """
            Generates one prediction per row, taking in a list of ordered dictionaries (one per row).
            """
            pd_df = pd.DataFrame.from_dict(feature_matrix)
            preds = self._model.predict_proba(pd_df)[:, 1]
            return preds

        def _get_required_features(self) -> OrderedSet:
            """
            An ordered set of feature column names
            """
            return self._required_features

    row = dataset.filter(dataset.row_id == 'xxxx').rdd.first()
    shparkley_wrapped_model = MyShparkleyModel(my_model)

    # You need to sample your dataset based on convergence criteria.
    # More samples results in more accurate shapley values.
    # Repartitioning and caching the sampled dataframe will speed up computation.
    sampled_df = training_df.sample(0.1, True).repartition(75).cache()

    shapley_scores_by_feature = compute_shapley_for_sample(
        df=sampled_df,
        model=shparkley_wrapped_model,
        row_to_investigate=row,
        weight_col_name='training_weight_column_name'
    )

