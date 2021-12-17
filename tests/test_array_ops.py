import unittest
import paoding.utility.array_operation as array_operation
import numpy as np

class TestArrayOperations(unittest.TestCase):

    def test_00_array_equal(self):
        
        sample_array = np.array([[0,0,1,2,3,4,5,6],
                          [1,0,1,2,3,4,5,6],
                          [2,0,1,2,3,4,5,6]])
        
        list0 = np.array([0,1,2,3,4,6])

        updated_list = array_operation.remove_from_array(array_operation.remove_from_array(sample_array[0], 6), 1)
        assert np.array_equal(updated_list, list0), "Error occurred in array deletion"

    def test_01_array_remove_row(self):
        
        sample_array = np.array([[0,0,1,2,3,4,5,6],
                          [1,0,1,2,3,4,5,6],
                          [2,0,1,2,3,4,5,6]])
        
        sample_array_rm_row = np.array([[0,0,1,2,3,4,5,6],
                            [2,0,1,2,3,4,5,6]])

        list1 = array_operation.remove_row_from_2d_array(sample_array, 1)
        assert np.array_equal(list1, sample_array_rm_row), "Error occurred in row deletion"

    def test_02_array_remove_col(self):
        
        sample_array = np.array([[0,0,1,2,3,4,5,6],
                          [1,0,1,2,3,4,5,6],
                          [2,0,1,2,3,4,5,6]])

        sample_array_rm_column = np.array([[0,0,1,2,4,5,6],
                          [1,0,1,2,4,5,6],
                          [2,0,1,2,4,5,6]])

        list2 = array_operation.remove_column_from_2d_array(sample_array, 4)
        assert np.array_equal(list2, sample_array_rm_column), "Error occurred in column deletion"
