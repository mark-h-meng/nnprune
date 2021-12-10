
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
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

# Import own classes
from nnprune.sampler import Sampler
from nnprune.evaluator import Evaluator
from nnprune.utility.option import SamplingMode, ModelType
import nnprune.utility.adversarial_mnist_fgsm_batch as adversarial
import nnprune.utility.training_from_data as training_from_data
import nnprune.utility.pruning as pruning
import nnprune.utility.utils as utils
import nnprune.utility.bcolors as bcolors
import nnprune.utility.interval_arithmetic as ia
import nnprune.utility.simulated_propagation as simprop

class Pruner:

    constant = 0
    model = None
    optimizer = None
    sampler = None
    robustness_evaluator = None
    model_path = None
    test_set = None
    BENCHMARKING_MODE = False
    
    first_mlp_layer_size = 0 
    model_type = -1

    lo_bound = 0
    hi_bound = 1
    
    def __init__(self, path, test_set, sample_strategy=None, alpha=0.75, input_interval=(0,1), first_mlp_layer_size=128, model_type=ModelType.XRAY):
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
        if sample_strategy == None:
            self.sampler = Sampler()
        else:
            self.sampler = sample_strategy
        self.robustness_evaluator = Evaluator()
        self.model_path = path
        # Specify a random seed
        seed(42)
        tf.random.set_seed(42)

        self.model_type = model_type

        self.BATCH_SIZE_PER_PRUNING = 4
        self.TARGET_ADV_EPSILONS = [0.5]
        self.POOLING_MULTIPLIER = 2
        self.TARGET_PRUNING_PERCENTAGE = 0.8
        self.BATCH_SIZE_PER_EVALUATION = 50

        self.TRAIN_BIGGER_MODEL = True

        # Specify the mode of pruning
        self.BASELINE_MODE = False

        # Recursive mode
        # PS: Baseline should is written in non-recursive mode
        self.RECURSIVE_PRUNING = False

        # E.g. EPOCHS_PER_CHECKPOINT = 5 means we save the pruned model as a checkpoint after each five
        #    epochs and at the end of pruning
        self.EPOCHS_PER_CHECKPOINT = 15
            
        hyper_parameter_alpha = alpha

        hyper_parameter_beta = 1-hyper_parameter_alpha
        self.hyperparameters = (hyper_parameter_alpha, hyper_parameter_beta)
        
        self.test_set = test_set

        (self.lo_bound, self.hi_bound) = input_interval
        self.first_mlp_layer_size = first_mlp_layer_size

    def load_model(self, optimizer=None):
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model.summary())
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        else:
            self.optimizer = optimizer

    def save_model(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
            print("Overwriting existing pruned model ...")

        self.model.save(path)
        print(" >>> Pruned model saved")
       
    def evaluate(self):
        test_features, test_labels = self.test_set
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        loss, accuracy = self.model.evaluate(test_features, test_labels, verbose=2)
        print("Evaluation accomplished -- [ACC]", accuracy, "[LOSS]", loss)   
        return loss, accuracy
    
    def evaluate_robustness(self):
        test_features, test_labels = self.test_set
        if self.model_type == ModelType.XRAY:
            robust_preservation = adversarial.robustness_evaluation_chest(self.model,
                                                                (test_features, test_labels),
                                                                self.TARGET_ADV_EPSILONS,
                                                                self.BATCH_SIZE_PER_EVALUATION)
        elif self.model_type == ModelType.CREDIT:
            robust_preservation = adversarial.robustness_evaluation_kaggle(self.model,
                                                                (test_features, test_labels),
                                                                self.TARGET_ADV_EPSILONS,
                                                                self.BATCH_SIZE_PER_EVALUATION)

                
        else:
            print("Robustness evaluation not available for this release!")
        return robust_preservation


    def prune(self, evaluate=False):
        if evaluate:
            self.BENCHMARKING_MODE = False
        else:
            self.BENCHMARKING_MODE = True
        test_images, test_labels = self.test_set
        utils.create_dir_if_not_exist("nnprune/logs/")
        utils.create_dir_if_not_exist("nnprune/save_figs/")
        
        pruned_model_path=self.model_path+"_pruned"

        # Define a list to record each pruning decision
        tape_of_moves = []
        # Define a list to record benchmark & evaluation per pruning epoch (begins with original model)
        score_board = []
        accuracy_board = []

        ################################################################
        # Launch a pruning epoch                                       #
        ################################################################

        epoch_couter = 0
        num_units_pruned = 0
        percentage_been_pruned = 0
        stop_condition = False
        neurons_manipulated =None
        target_scores = None
        pruned_pairs = None
        cumulative_impact_intervals = None
        saliency_matrix=None
        
        model = self.model

        big_map = simprop.get_definition_map(model, input_interval=(self.lo_bound, self.hi_bound))
    
        num_units_first_mlp_layer = self.first_mlp_layer_size
        # Start elapsed time counting
        start_time = time.time()

        while(not stop_condition):

            pruning_result_dict = self.sampler.prune(model,big_map, 
                                                prune_percentage=self.BATCH_SIZE_PER_PRUNING/num_units_first_mlp_layer,
                                                cumulative_impact_intervals=cumulative_impact_intervals,
                                                pooling_multiplier=self.POOLING_MULTIPLIER,
                                                hyperparamters=self.hyperparameters,
                                                neurons_manipulated=neurons_manipulated, saliency_matrix=saliency_matrix,
                                                recursive_pruning=False,bias_aware=True)

            model = pruning_result_dict['model']
            neurons_manipulated = pruning_result_dict['neurons_manipulated']
            target_scores = pruning_result_dict['target_scores']
            pruned_pairs = pruning_result_dict['pruned_pairs']
            cumulative_impact_intervals = pruning_result_dict['cumulative_impact_intervals']
            saliency_matrix = pruning_result_dict['saliency_matrix']
            score_dicts = pruning_result_dict['pruning_pairs_dict_overall_scores']

            epoch_couter += 1

            # Check if the list of pruned pair is empty or not - empty means no more pruning is feasible
            num_pruned_curr_batch = 0
            if pruned_pairs is not None:
                for layer, pairs in enumerate(pruned_pairs):
                    if len(pairs) > 0:
                        num_pruned_curr_batch += len(pairs)

            if num_pruned_curr_batch == 0:
                stop_condition = True
                print(" >> No more hidden unit could be pruned, we stop at EPOCH", epoch_couter)
            else:
                if not self.sampler.mode == SamplingMode.BASELINE:
                    print(" >> Cumulative impact as intervals after this epoch:")
                    print(cumulative_impact_intervals)

                percentage_been_pruned += self.BATCH_SIZE_PER_PRUNING/num_units_first_mlp_layer
                print(" >> Pruning progress:", bcolors.BOLD, str(percentage_been_pruned * 100) + "%", bcolors.ENDC)

                model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
                if not self.BENCHMARKING_MODE:
                    '''
                    robust_preservation = adversarial.robustness_evaluation_chest(model,
                                                                            (test_images, test_labels),
                                                                            self.TARGET_ADV_EPSILONS,
                                                                            self.BATCH_SIZE_PER_EVALUATION)
                    '''
                    robust_preservation = self.evaluate_robustness()
                    #loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
                    loss, accuracy = self.evaluate()

                    # Update score_board and tape_of_moves
                    score_board.append(robust_preservation)
                    accuracy_board.append((round(loss, 4), round(accuracy, 4)))
                    print(bcolors.OKGREEN + "[Epoch " + str(epoch_couter) + "]" + str(robust_preservation) + bcolors.ENDC)

                tape_of_moves.append(pruned_pairs)
                pruned_pairs = None
            # Check if have pruned enough number of hidden units
            if self.sampler.mode == SamplingMode.BASELINE and percentage_been_pruned >= 0.5:
                print(" >> Maximum pruning percentage has been reached")
                stop_condition = True
            elif not stop_condition and percentage_been_pruned >= self.TARGET_PRUNING_PERCENTAGE:
                print(" >> Target pruning percentage has been reached")
                stop_condition = True

            # Save the pruned model at each checkpoint or after the last pruning epoch
            if epoch_couter % self.EPOCHS_PER_CHECKPOINT == 0 or stop_condition:
                curr_pruned_model_path = pruned_model_path + "_ckpt_" + str(self.hyperparameters[0]) + "_" + str(math.ceil(epoch_couter/self.EPOCHS_PER_CHECKPOINT))

                if os.path.exists(curr_pruned_model_path):
                    shutil.rmtree(curr_pruned_model_path)
                print("Overwriting existing pruned model ...")

                model.save(curr_pruned_model_path)
                print(" >>> Pruned model saved")

        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        ################################################################
        # Save the tape of moves                                       #
        ################################################################
        
        # Obtain a timestamp
        local_time = time.localtime()
        timestamp = time.strftime('%b-%d-%H%M', local_time)


        tape_filename = "nnprune/logs/chest-" + timestamp + "-" + str(self.BATCH_SIZE_PER_EVALUATION)
        if self.BENCHMARKING_MODE:
            tape_filename = tape_filename+"-BENCHMARK"

        if self.sampler.mode == SamplingMode.BASELINE:
            tape_filename += "_tape_baseline.csv"
        else:
            tape_filename = tape_filename + "_tape_" + self.sampler.mode.name + "_" + str(self.hyperparameters[0]) + ".csv"

        if os.path.exists(tape_filename):
            os.remove(tape_filename)

        with open(tape_filename, 'w+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_line = [str(eps) for eps in self.TARGET_ADV_EPSILONS]
            csv_line.append('moves,loss,accuracy')
            csv_writer.writerow(csv_line)

            for index, item in enumerate(score_board):
                rob_pres_stat = [item[k] for k in self.TARGET_ADV_EPSILONS]
                rob_pres_stat.append(tape_of_moves[index])
                rob_pres_stat.append(accuracy_board[index])
                csv_writer.writerow(rob_pres_stat)
            
            if self.BENCHMARKING_MODE:
                csv_writer.writerow(["Elapsed time: ", round((end_time - start_time) / 60.0, 3), "minutes /", int(end_time - start_time), "seconds"])

        print("Pruning accomplished")
 