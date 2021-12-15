
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

from paoding.utility.option import ModelType, AttackAlogirithm
import paoding.utility.adversarial_mnist_fgsm_batch as adversarial

class Evaluator:

    epsilons = []
    batch_size = 0
    attack_mode = -1
    metrics = ['accuracy']

    def __init__(self, epsilons = [0.5], batch_size = 50, attack_mode = AttackAlogirithm.FGSM, k=1):
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
        if type(epsilons) == list:
            self.epsilons = epsilons
        else:
            self.epsilons = [epsilons]
        self.batch_size = batch_size
        self.attack_mode = attack_mode

        if k > 1:
            self.metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k))

    def evaluate_robustness(self, model, test_set, model_type, k=1):
        if self.attack_mode == AttackAlogirithm.FGSM:
            return self.__fgsm(model, test_set, model_type, k)
        else:
            print("Evaluation mode not set or set to an illegal value, please check!")
        
    def __fgsm(self, model, test_set, model_type, k):
        test_features, test_labels = test_set
        if model_type == ModelType.XRAY:
            robust_preservation = adversarial.robustness_evaluation_chest(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.CREDIT:
            robust_preservation = adversarial.robustness_evaluation_kaggle(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.MNIST:
            robust_preservation = adversarial.robustness_evaluation(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        elif model_type == ModelType.CIFAR:
            if k > 1:
                robust_preservation = adversarial.robustness_evaluation_cifar_topK(model, 
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size, k)

            else:
                robust_preservation = adversarial.robustness_evaluation_cifar(model,
                                                                (test_features, test_labels),
                                                                self.epsilons,
                                                                self.batch_size)
        else:
            print("Robustness evaluation not available for this release!")
        return robust_preservation

