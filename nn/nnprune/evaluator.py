
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

class Evaluator:

    constant = 100

    def __init__(self):
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
        pass

    def __generate_adversarial_samples(self):
        print("Generating adversarial samplers.")

    def __fgsm(self):
        print("FGSM gradient computed.")

    def evaluate(self):
        print("Evaluation result generated.")
