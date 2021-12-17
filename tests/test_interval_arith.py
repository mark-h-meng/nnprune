import unittest
import paoding.utility.interval_arithmetic as interval_arithmetic


class TestIntervalArithmetic(unittest.TestCase):
    
    def test_00_basic_arithmetic(self):
        
        (a, b) = (-0.9, 0.9)
        (c, d) = (-0.1, 0.1)
        (e, f) = (0.1, 0.9)


        assert (-1.0, 1.0) == interval_arithmetic.interval_sum([(a, b), (c, d)]), "Error occurred in interval sum operation"
        assert (-0.8, 1.8) == interval_arithmetic.interval_add((a, b), (e, f)), "Error occurred in interval minus operation"
        assert (-1.0, 1.0) == interval_arithmetic.interval_minus((a, b), (c, d)), "Error occurred in interval minus operation"
        assert (-1.0, 1.0) == interval_arithmetic.interval_scale((c, d), 10), "Error occurred in interval scale operation"

    def test_01_forward_propagation(self):

        (a, b) = (-0.9, 0.9)
        (c, d) = (-0.1, 0.1)
        (e, f) = (0.1, 0.9)

        b_params = [-1, 1]
        w_params = [[0.5, -0.5],
                    [0.5, -0.5],
                    [0.5, -0.5]]

        res_fp = interval_arithmetic.forward_propogation([(a,b), (c,d), (e,f)], w_params, b_params, activation=False, relu_activation=False)
        res_fp_testing = []

        for i in res_fp:
            (l, h) = i
            l = round(l,2)
            h = round(h,2)
            res_fp_testing.append((l,h))

        assert [(-1.45, -0.05), (0.05, 1.45)] == res_fp_testing, "Error occurred in forward propagation"

    def test_02_budget_handling(self):
        budget = [(-0.5, 0.5),
                (-0.5, 0.5),
                (-0.5, 0.5)]

        utilized_budget = [(-0.4, 0.4),
                        (-0.5, 0.5),
                        (0, 0.4)]

        assert interval_arithmetic.check_budget_preservation(utilized_budget, budget) == [1, 0, 1], \
            "Error occurred in checking budget preservation"

        assert interval_arithmetic.interval_list_add(utilized_budget, budget) == [(-0.9, 0.9), (-1.0, 1.0), (-0.5, 0.9)] , \
            "Error occurred in adding two interval lists"


if __name__ == "__main__":
    unittest.main()