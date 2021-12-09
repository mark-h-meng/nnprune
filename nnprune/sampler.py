
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
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from numpy.random import seed
import os, time, csv, sys, shutil, math, time

from tensorflow.python.eager.monitoring import Sampler

# Import own classes
from nnprune.sampler import Sampler
from nnprune.evaluator import Evaluator
from nnprune.utility.option import SamplingMode
import nnprune.utility.adversarial_mnist_fgsm_batch as adversarial
import nnprune.utility.training_from_data as training_from_data
import nnprune.utility.pruning as pruning
import nnprune.utility.utils as utils
import nnprune.utility.bcolors as bcolors
import nnprune.utility.interval_arithmetic as ia
import nnprune.utility.simulated_propagation as simprop

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

    def nominate_candidate(self):
        print("Candidates to be pruned nominated.")