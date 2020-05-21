Shparkley: Scaling Shapley Values with Spark
=============================================

.. inclusion-marker-start-do-not-remove

.. contents::

Shparkley is a PySpark implementation of
`Shapley values <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf>`_
which uses a `monte-carlo approximation <https://link.springer.com/article/10.1007/s10115-013-0679-x>`_ algorithm.

Given a dataset and machine learning model, Shparkley can compute Shapley values for all features for a feature vector.
Shparkley also handles training weights and is model agnostic.

Installation
------------

``pip install shparkley``

Requirements
------------
You must have Apache Spark installed on your machine/cluster.


Example Usage
--------------

.. code-block:: python

    class MyShparkleyModel(ShparkleyModel):
    """
    You need to wrap your model with the ShparkleyModel interface.
    """
        def get_required_features(self):
            # type: () -> Set[str]
            """
            Needs to return a set of feature names for the model.
            """
            return ['feature-1', 'feature-2', 'feature-3']

        def predict(self, feature_matrix):
            # type: (List[Dict[str, Any]]) -> List[float]
            """
            Wrapper function to convert the feature matrix into an acceptable format for your model.
            This function should return the predicted probabilities.
            The feature_matrix is a list of feature dictionaries.
            Each dictionary has a mapping from the feature name to the value.
            :return: Model predictions for all feature vectors
            """
            # Convert the feature matrix into an appropriate form for your model object.
            pd_df = pd.DataFrame.from_dict(feature_matrix)
            preds = self._model.my_predict(pd_df)
            return preds

    row = dataset.filter(dataset.row_id = 'xxxx').rdd.first()
    shparkley_wrapped_model = MyShparkleyModel(my_model)

    # You need to sample your dataset based on convergence criteria.
    # More samples results in more accurate shapley values.
    # Repartitioning and caching the sampled dataframe will speed up computation.
    sampled_df = posv7_df.sample(0.1, True).repartition(75).cache()

    shapley_scores_by_feature = compute_shapley_for_sample(
        df=sampled_df,
        model=shparkley_wrapped_model,
        row_to_investigate=row,
        weight_col_name='training_weight_column_name'
    )

