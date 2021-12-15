
# Copyright 2021 Mark H. Meng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/python3

# Import publicly published & installed packages
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from numpy.random import seed
import os, time, csv, sys, shutil, math, time

from tensorflow.python.eager.monitoring import Sampler

# Import own classes
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
from paoding.utility.option import SamplingMode
import paoding.utility.adversarial_mnist_fgsm_batch as adversarial
import paoding.utility.training_from_data as training_from_data
import paoding.utility.pruning as pruning
import paoding.utility.utils as utils
import paoding.utility.bcolors as bcolors
import paoding.utility.interval_arithmetic as ia
import paoding.utility.simulated_propagation as simprop

class Sampler:

    mode = -1

    def __init__(self, mode=SamplingMode.BASELINE):
        """Initializes `Loss` class.
        Args:
        reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
        name: Optional name for the instance.
        """
        self.mode = mode
        pass

    def __saliency_based_sampler(self):
        print("Saliency-based sampling (baseline) selected.")

    def __greedy_sampler(self):
        print("Greedy sampling selected.")

    def __stochastic_sampler(self):
        print("Stochastic sampling selected.")

    def prune(self, model, big_map, prune_percentage=None,
                     neurons_manipulated=None, saliency_matrix=None,
                     recursive_pruning=False, cumulative_impact_intervals=None,
                     bias_aware=False, pooling_multiplier=1,
                     target_scores=None, hyperparamters=(0.5, 0.5)):
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
                   hyperparamters,
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
                      hyperparamters,
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
