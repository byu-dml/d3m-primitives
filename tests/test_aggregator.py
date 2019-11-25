import unittest

import pandas as pd
import numpy as np
from d3m import container

from byudml.aggregator.aggregator import Aggregator
from tests.utils import make_df, print_df, are_df_elements_equal, Column, ATTRIBUTE, PREDICTED_TARGET

class TestAggregatorPrimitive(unittest.TestCase):

    def setUp(self):
        self.hyperparams_class = Aggregator.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self.float_df = make_df([
            Column('floata', [1.0, 1.0, 2.0, 5.0], [ATTRIBUTE]),
            Column('floatb', [2.0, 3.0, 2.0, 7.0], [ATTRIBUTE]),
            Column('floatc', [3.0, 5.0, 2.0, 10.5], [ATTRIBUTE]),
        ])
        self.numeric_df = make_df([
            Column('floata', [1.0, 1.0, 2.0, 1.0], [ATTRIBUTE]),
            Column('floatb', [5.0, 4.0, 3.0, 6.0], [ATTRIBUTE]),
            Column('inta', [3, 2, 1, 5], [ATTRIBUTE]),
            Column('boola', [True, False, True, True], [ATTRIBUTE])
        ])
        self.mixed_df = make_df([
            Column('float', [5.0, 3, 10], [ATTRIBUTE]),
            Column('cat', ['a', 'b', 'c'], [ATTRIBUTE]),
            Column('inta', [2, 1, 1], [ATTRIBUTE]),
            Column('intb', [2, 2, 1], [ATTRIBUTE]),
        ])
        self.one_column_df = make_df([
            Column('float', [3.3, 4.4, 5.5, 6.6], [ATTRIBUTE])
        ])
        self.numeric_with_nans_df = make_df([
            Column('floata', [1.0, 1.0, 1.0, np.nan], [ATTRIBUTE]),
            Column('floatb', [6.0, 5.0, np.nan, 6.0], [ATTRIBUTE]),
            Column('inta', [3, np.nan, 1, 5], [ATTRIBUTE]),
            # Note: Pandas cannot have a bool column with NaN values.
            # Pandas will make the column of type `object`.
            Column('boola', [False, False, True, True], [ATTRIBUTE])
        ])
        self.mixed_type_modes_df = make_df([
            Column('inta', [1 ,2, 3, 4], [ATTRIBUTE]),
            Column('intb', [1, 2, 4, 3], [ATTRIBUTE]),
            Column('cata', ['a', 'b', 'c', 'd'], [ATTRIBUTE]),
            Column('catb', ['b', 'a', 'c', 'd'], [ATTRIBUTE])
        ])
        self.categorical_df = make_df([
            Column('cata', ['a', 'b', 'b', 'a'] ,[ATTRIBUTE]),
            Column('catb', ['a', 'c', 'c', 'b'] ,[ATTRIBUTE]),
            Column('catc', ['a', 'c', 'b', 'a'] ,[ATTRIBUTE]),
        ])
    
    def test_can_take_mean(self) -> None:
        # Check the basic case: all floats.
        outputs = self._produce(self.float_df, aggregator='mean')
        expected_output = pd.DataFrame({'mean': [2.0, 3.0, 2.0, 7.5]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))
    
    def test_can_take_mean_of_all_numerics(self) -> None:
        # Check to make sure it can handle all numeric types (float, int, and bool).
        outputs = self._produce(self.numeric_df, aggregator='mean')
        expected_output = pd.DataFrame({'mean': [2.5, 1.75, 1.75, 3.25]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))
    
    def test_can_take_mean_of_all_non_numerics(self) -> None:
        outputs = self._produce(self.categorical_df, aggregator='mean')
        # The mean column will necessarily be all NaN values
        self.assertTrue(outputs.isnull().values.all())

    def test_mean_ignores_non_numeric(self) -> None:
        outputs = self._produce(self.mixed_df, aggregator='mean')
        expected_output = pd.DataFrame({'mean': [3.0, 2.0, 4.0]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))

    def test_mean_can_handle_one_column(self) -> None:
        outputs = self._produce(self.one_column_df, aggregator='mean')
        expected_output = pd.DataFrame({'mean': [3.3, 4.4, 5.5, 6.6]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))
    
    def test_mean_can_handle_nans(self) -> None:
        outputs = self._produce(self.numeric_with_nans_df, aggregator='mean')
        expected_output = pd.DataFrame({'mean': [2.5, 2.0, 1.0, 4.0]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))
    
    def test_can_take_mode(self) -> None:
        # Check the basic case: all floats.
        outputs = self._produce(self.float_df, aggregator='mode')

        self.assertEqual(outputs.shape, (4,1))
        self.assertIn(outputs.iloc[0,0], [1.0, 2.0, 3.0])
        self.assertIn(outputs.iloc[1,0], [1.0, 3.0, 5.0])
        self.assertEqual(outputs.iloc[2,0], 2.0)
        self.assertIn(outputs.iloc[3,0], [5.0, 7.0, 10.5])
    
    def test_can_take_mode_of_all_non_numerics(self) -> None:
        outputs = self._produce(self.categorical_df, aggregator='mode')
        expected_output = pd.DataFrame({'mode': ['a', 'c', 'b', 'a']})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))

    def test_mode_can_handle_one_column(self) -> None:
        outputs = self._produce(self.one_column_df, aggregator='mode')
        expected_output = pd.DataFrame({'mode': [3.3, 4.4, 5.5, 6.6]})
        self.assertTrue(are_df_elements_equal(outputs, expected_output))
    
    def test_mode_can_handle_nans(self) -> None:
        outputs = self._produce(self.numeric_with_nans_df, aggregator='mode')
        
        self.assertEqual(outputs.shape, (4,1))
        self.assertIn(outputs.iloc[0,0], [1.0, 6.0, 3.0, False])
        self.assertIn(outputs.iloc[1,0], [1.0, 5.0, False])
        self.assertEqual(outputs.iloc[2,0], 1.0)
        self.assertIn(outputs.iloc[3,0], [6.0, 5.0, True])

    def test_mode_can_handle_mode_col_of_mixed_types(self) -> None:
        outputs = self._produce(self.mixed_type_modes_df, aggregator='mode')
        expected_output = pd.DataFrame({'mode': [1, 2, 'c', 'd']})
        # Note: when the modes are mixed types, the pandas column will
        # necessarily be of type `object`.
        self.assertTrue(are_df_elements_equal(outputs, expected_output))

    def test_can_change_output_semantic_types(self) -> None:
        regular_outputs = self._produce(self.float_df, aggregator='mean')
        attr_cols = regular_outputs.metadata.list_columns_with_semantic_types([ATTRIBUTE])
        self.assertEqual(len(attr_cols), 1)

        predicted_outputs = self._produce(
            self.float_df,
            aggregator='mean',
            output_semantic_types=[PREDICTED_TARGET]
        )
        pred_cols = predicted_outputs.metadata.list_columns_with_semantic_types([PREDICTED_TARGET])
        self.assertEqual(len(pred_cols), 1)

    def _produce(self, inputs: container.DataFrame, **hyperparams) -> container.DataFrame:
        prim = Aggregator(hyperparams=self.hyperparams_class.defaults().replace(hyperparams))
        return prim.produce(inputs=inputs).value
