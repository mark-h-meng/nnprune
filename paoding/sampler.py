#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf
import time

# Import in-house classes
from paoding.utility.option import SamplingMode
import paoding.utility.pruning as pruning

class Sampler:

    mode = -1
    params=(0, 0)

    def __init__(self, mode=SamplingMode.BASELINE, params=(0.75, 0.25)):
        """Initializes `Sampler` class.
        Args:
        mode: The mode of sampling strategy (optional, baseline mode by default).
            [PS] 3 modes are supported in the Alpha release, refer to the ``paoding.utility.option.SamplingMode`` for the technical definition.
        params: The tuple of parameters (for greedy and stochastic modes only) (optional, (0.75, 0.25) by default).
        """
        self.mode = mode
        self.params = params
        pass

    def nominate(self, model, big_map, prune_percentage=None,
                     neurons_manipulated=None, saliency_matrix=None,
                     recursive_pruning=False, cumulative_impact_intervals=None,
                     bias_aware=False, pooling_multiplier=2,
                     target_scores=None):
        """
        TO-DO
        Initializes `Sampler` class.
        Args:
        mode: The mode of sampling strategy (optional, baseline mode by default).
            [PS] 3 modes are supported in the Alpha release, refer to the ``paoding.utility.option.SamplingMode`` for the technical definition.
        """
        pruned_pairs = None
        pruning_pairs_dict_overall_scores = None
        if self.mode == SamplingMode.BASELINE:
            result = pruning.pruning_baseline(model, big_map, prune_percentage, neurons_manipulated,
                                        saliency_matrix, recursive_pruning, bias_aware)
            (model, neurons_manipulated, pruned_pairs, saliency_matrix) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1

        elif self.mode == SamplingMode.GREEDY:
            result = pruning.pruning_greedy(model, big_map, prune_percentage,
                   cumulative_impact_intervals,
                   pooling_multiplier,
                   neurons_manipulated,
                   self.params,
                   recursive_pruning,
                   bias_aware,
                   kaggle_credit=False)
            (model, neurons_manipulated, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        print(" >>   with assessment score ", end=' ')
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1
                            print(round(pruning_pairs_dict_overall_scores[layer][pair], 3), end=' ')
                        print()

        elif self.mode == SamplingMode.STOCHASTIC:
            result = pruning.pruning_stochastic(model, big_map, prune_percentage,
                      cumulative_impact_intervals,
                      neurons_manipulated,
                      target_scores,
                      self.params,
                      recursive_pruning,
                      bias_aware,
                      kaggle_credit=False)
            (model, neurons_manipulated, target_scores, pruned_pairs, cumulative_impact_intervals, pruning_pairs_dict_overall_scores) = result

            count_pairs_pruned_curr_epoch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        print(" >> Pruning", pairs, "at layer", str(layer))
                        print(" >>   with assessment score ", end=' ')
                        for pair in pairs:
                            count_pairs_pruned_curr_epoch += 1
                            print(round(pruning_pairs_dict_overall_scores[layer][pair], 3), end=' ')
                        print()
                        print(" >> Updated target scores at this layer:", round(target_scores[layer], 3))
        else:
            print("Mode not recognized, execution aborted!")
        
        result_dict = {
            'model': model,
            'neurons_manipulated': neurons_manipulated,
            'target_scores': target_scores,
            'pruned_pairs': pruned_pairs,
            'saliency_matrix': saliency_matrix,
            'cumulative_impact_intervals': cumulative_impact_intervals,
            'pruning_pairs_dict_overall_scores': pruning_pairs_dict_overall_scores
        }

        return result_dict
