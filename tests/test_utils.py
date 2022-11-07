import unittest
import paoding.utility.utils as utils
import pandas as pd

class TestUtils(unittest.TestCase):

    def test_redundant_pairs(self):
        df_test = pd.DataFrame(columns=["var1", "var2", "var3"], data=[
            [1, 0.8, 0.8],
            [0.6, 1, 0],
            [0.8, 0, 1]
        ])
        expected = {('var1', 'var1'), ('var2', 'var2'), ('var3', 'var3')}
        generated = utils.get_redundant_pairs(df_test)
        assert (generated == expected)


    def test_l1_norm_of_intervals(self):
        interval_list_test = [(0,1),(1,3),(3,4),(5,7),(9,100)]
        expected = 1 + 2 + 1 + 2 + 91
        generated = utils.l1_norm_of_intervals(interval_list_test)
        assert (generated == expected)

    def test_union_of_interval(self):
        interval_list_test = [(0,1),(1,3),(3,4),(5,7),(9,100)]
        expected = (0, 100)
        generated = utils.union_of_interval(interval_list_test)
        assert (generated == expected)

    def test_calculate_similarity_of_two_intervals(self):
        interval_test_1 = (0,1)
        interval_test_2 = (-1,1)
        u_interval = utils.union_of_interval([interval_test_1, interval_test_2])
        expected = 0.75
        generated = utils.calculate_similarity_of_two_intervals(interval_test_1, interval_test_2, u_interval)

        assert (expected == generated)

if __name__ == '__main__':
    unittest.main()