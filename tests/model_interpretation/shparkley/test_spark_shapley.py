from __future__ import absolute_import
from __future__ import division
import unittest
import pandas as pd
from collections import Counter
from mock import MagicMock
import pyspark.sql
from pyspark.sql import Row
from affirm.model_interpretation.shparkley.spark_shapley import (
    compute_shapley_score,
    compute_shapley_for_sample,
    ShparkleyModel
)


class TestShparkleyModel(ShparkleyModel):

    def get_required_features(self):
        return self._model.get_required_features()

    def predict(self, feature_matrix):
        pd_df = pd.DataFrame.from_dict(feature_matrix)
        preds = self._model.predict(pd_df)
        return preds


def model_predict_side_effect_function(df):
    df['score'] = (df['f1'] * 3 + df['f2'] * 5)
    return df['score'].values


class SparkShapleyTest(unittest.TestCase):

    def setUp(self):
        self.m_model = MagicMock()
        self.m_model.get_required_features.return_value = ['f1', 'f2']
        self.m_model.predict.side_effect = model_predict_side_effect_function
        self.row1 = {
            'f1': 0.01,
            'f2': 0.05,
        }
        self.row2 = {
            'f1': 0.2,
            'f2': 0.5,
        }
        self.row3 = {
            'f1': 1.0,
            'f2': 0.5,
        }
        self.row_investigate = {
            'f1': 7.0,
            'f2': 6.5,
        }
        self.m_shparkley_model = TestShparkleyModel(self.m_model)
        builder = pyspark.sql.SparkSession.builder.master('local[1]')
        self.spark = builder.appName('unittest').getOrCreate()

    def tearDown(self):
        if self.spark is not None:
            self.spark.stop()

    def test_compute_shapley_for_sample(self):
        dataset = self.spark.createDataFrame([self.row1, self.row2, self.row3])
        shapley_scores = compute_shapley_for_sample(
            df=dataset,
            model=self.m_shparkley_model,
            row_to_investigate=Row(**self.row_investigate)
        )
        sorted_shapley_scores = sorted([(k, v) for k, v in shapley_scores.items()])
        self.assertEquals(sorted_shapley_scores, [('f1', 19.79), ('f2', 30.75)])

    def test_compute_shapley_for_sample_weighted(self):

        # Add weights: [2, 1, 1]
        self.row1.update({'weight': 2})
        for row in (self.row2, self.row3):
            row.update({'weight': 1})

        dataset = self.spark.createDataFrame([self.row1, self.row2, self.row3])
        shapley_scores = compute_shapley_for_sample(
            df=dataset,
            model=self.m_shparkley_model,
            row_to_investigate=Row(**self.row_investigate),
            weight_col_name='weight',
        )

        sorted_shapley_scores = sorted([(k, v) for k, v in shapley_scores.items()])
        # row 1, which has small feature vals (=> smaller prediction) weighted more heavily means that compared to
        # baseline, the relatively large features in the sample of interest (=> bigger prediction => more different
        # from row 1) will increase both shapley values to the below compared to unweighted (19.79, 30.75)
        self.assertEquals(sorted_shapley_scores, [('f1', 20.085), ('f2', 31.125)])

    def test_compute_shapley_score(self):
        row_samples = [Row(**self.row1), Row(**self.row2), Row(**self.row3)]
        scores = compute_shapley_score(
            partition_index=1,
            rand_rows=row_samples,
            row_to_investigate=Row(**self.row_investigate),
            model=self.m_shparkley_model
        )
        expected_result = [
            ('f1', 18.0, 1.0),
            ('f1', 20.4, 1.0),
            ('f1', 20.97, 1.0),
            ('f2', 30.0, 1.0),
            ('f2', 30.0, 1.0),
            ('f2', 32.25, 1.0)
        ]
        self.assertEqual(sorted(list(scores)), expected_result)

    def test_compute_shapley_score_weighted(self):

        # Get unweighted result for reference
        row_samples = [Row(**self.row1), Row(**self.row2), Row(**self.row3)]
        unweighted_result = compute_shapley_score(
            partition_index=1,
            rand_rows=row_samples,
            row_to_investigate=Row(**self.row_investigate),
            model=self.m_shparkley_model
        )

        unweighted_result = sorted(list(unweighted_result))

        # Add weights: [2, 1, 1]
        self.row1.update({'weight': 2})
        for row in (self.row2, self.row3):
            row.update({'weight': 1})
        row_samples__weighted_first_double = [Row(**self.row1), Row(**self.row2), Row(**self.row3)]

        weighted_first_double_result = compute_shapley_score(
            partition_index=1,
            rand_rows=row_samples__weighted_first_double,
            row_to_investigate=Row(**self.row_investigate),
            model=self.m_shparkley_model,
            weight_col_name='weight',
        )

        weights_found = Counter()
        for (ft_name_uw, ft_val_uw, ft_val_weight_uw), (ft_name_double, ft_val_double, ft_val_weight) in zip(
                unweighted_result,
                sorted(list(weighted_first_double_result))
        ):
            self.assertEqual(ft_name_uw, ft_name_double)
            self.assertEqual(ft_val_uw, ft_val_double)
            weights_found[ft_val_weight] += 1

        # Four (two rows times two features) of \phi_j^ms of weight 1, two (one rwo times two features) of
        # weight 2
        self.assertDictEqual(weights_found, {1.0: 4, 2.0: 2})

    def test_efficiency_property(self):
        # The Shapley value must satisfy the Efficiency property
        # The feature contributions must add up to the difference of prediction for x and the average.
        dataset = self.spark.createDataFrame([self.row1, self.row2, self.row3])
        shapley_scores = compute_shapley_for_sample(
            df=dataset,
            model=self.m_shparkley_model,
            row_to_investigate=Row(**self.row_investigate)
        )
        total_shapley_value = sum([v for _, v in shapley_scores.items()])
        predicted_value_for_row = model_predict_side_effect_function(
            pd.DataFrame.from_dict([self.row_investigate])
        )
        rows = [self.row1, self.row2, self.row3]
        scores = model_predict_side_effect_function(pd.DataFrame.from_dict(rows))
        mean_prediction_on_dataset = sum(scores)/len(rows)
        self.assertAlmostEquals(
            first=total_shapley_value,
            second=predicted_value_for_row - mean_prediction_on_dataset,
            delta=0.01
        )
