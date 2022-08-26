#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2021, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"

# Import publicly published & installed packages
import tensorflow as tf
from numpy.random import seed
import os, time, csv, shutil, math, time
from pathlib import Path

# Import in-house classes
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
from paoding.utility.option import SamplingMode, ModelType
import paoding.utility.utils as utils
import paoding.utility.bcolors as bcolors
import paoding.utility.simulated_propagation as simprop
import paoding.utility.model_profiler.profiler as profiler

class Pruner:

    constant = 0
    model = None
    optimizer = None
    sampler = None
    robustness_evaluator = None
    model_path = None
    test_set = None

    pruning_target = None
    pruning_step = None
    
    model_type = -1

    lo_bound = 0
    hi_bound = 1
    
    def __init__(self, path, test_set=None, target=0.5, step=0.025, sample_strategy=None, input_interval=(0,1), model_type=ModelType.XRAY, seed_val=None):
        """
        Initializes `Pruner` class.
        Args:     
        path: The path of neural network model to be pruned.
        test_set: The tuple of test features and labels used for evaluation purpose.
        target: The percentage value of expected pruning goal (optional, 0.50 by default).
        step: The percentage value of pruning portion during each epoch (optional, 0.025 by default).
        sample_strategy: The sampling strategy specified for pruning (optional).
        alpha: The value of alpha parameters to be used in stochastic mode (optional, 0.75 by default).
        input_interval: The value range of an legal input (optional, [0,1] by default).
        model_type: The enumerated value that specifies the model type (optional, binary classification model by default).
            [PS] 4 modes are supported in the Alpha release, refer to the ``paoding.utility.option.ModelType`` for the technical definition.
        seed: The seed for randomization for the reproducibility purpose (optional, to use only for the experimental purpose)
        """
        if sample_strategy == None:
            self.sampler = Sampler()
        else:
            self.sampler = sample_strategy
        self.robustness_evaluator = Evaluator()
        
        self.model_path = path
        # Specify a random seed
        if seed_val is not None:
            seed(seed_val)
            tf.random.set_seed(seed_val)

        self.model_type = model_type

        self.target_adv_epsilons = [0.5]

        self.pruning_target = target
        self.pruning_step = step
        
        self.evaluation_batch = 50

        # E.g. EPOCHS_PER_CHECKPOINT = 5 means we save the pruned model as a checkpoint after each five
        #    epochs and at the end of pruning
        self.EPOCHS_PER_CHECKPOINT = 1000
        
        self.test_set = test_set

        (self.lo_bound, self.hi_bound) = input_interval
        #self.first_mlp_layer_size = first_mlp_layer_size

    def load_model(self, optimizer=None):
        """
        Load the model.
        Args: 
        optimizer: The optimizer specified for evaluation purpose (optional, RMSprop with lr=0.01 by default).
        """
        self.model = tf.keras.models.load_model(self.model_path)
        print(self.model.summary())
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            #self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)
        else:
            self.optimizer = optimizer

    def save_model(self, path):
        """
        Save the model to the path specified.
        Args: 
        path: The path that the model to be saved.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
            print("Overwriting existing pruned model ...")

        self.model.save(path)
        print(" >>> Pruned model saved")
       
    def evaluate(self, verbose=0):
        """
        Evaluate the model performance.
        Args: 
        metrics: The list of TF compatible metrics (optional, accuracy (only) by default).
        Returns:
        A tuple of loss and accuracy values
        """
        if self.test_set is None:
            print("Test set not provided, evaluation aborted...")
            return 0, 0

        test_features, test_labels = self.test_set
        # self.model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=metrics)
        loss, accuracy = self.model.evaluate(test_features, test_labels, verbose=2)
        if verbose > 0:
            print("Evaluation accomplished -- [ACC]", accuracy, "[LOSS]", loss)   
        return loss, accuracy


    def profile(self):
        print(profiler.model_profiler(self.model, Batch_size=1))

    def quantization(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        tflite_models_dir = Path('paoding/models/temp_tflite_models/')
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_file = tflite_models_dir/"model.tflite"  
        tflite_model_file.write_bytes(tflite_model)
        print(" >> Size after pruning:", os.path.getsize(tflite_model_file))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16_model = converter.convert()
        tflite_model_fp16_file = tflite_models_dir/"model_quant_f16.tflite"
        tflite_model_fp16_file.write_bytes(tflite_fp16_model)
        print(" >> Size after quantization:", os.path.getsize(tflite_model_fp16_file))

    def prune(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0):
        """
        Perform fully connected pruning and save the pruned model to a specified location.
        Args: 
        evaluator: The evaluation configuration (optional, no evaluation requested by default).
        pruned_model_path: The location to save the pruned model (optional, a fixed path by default).
        """
        self.prune_fc(evaluator, save_file, pruned_model_path, verbose)
        self.prune_cnv(evaluator, save_file, pruned_model_path, verbose)

    def prune_fc(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0, model_name=None):
        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size

        if model_name is None:
            model_name = self.model_type.name

        test_images, test_labels = self.test_set
        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if save_file and pruned_model_path is None:
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
    
        # Start elapsed time counting
        start_time = time.time()

        while(not stop_condition):

            pruning_result_dict = self.sampler.nominate(model,big_map, 
                                                prune_percentage=self.pruning_step,
                                                cumulative_impact_intervals=cumulative_impact_intervals,
                                                neurons_manipulated=neurons_manipulated, saliency_matrix=saliency_matrix,
                                                bias_aware=True)

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
                if verbose > 0:
                    print(" [DEBUG] No more hidden unit could be pruned, we stop at EPOCH", epoch_couter)
            else:
                if not self.sampler.mode == SamplingMode.BASELINE:
                    if verbose > 0:
                        print(" [DEBUG] Cumulative impact as intervals after this epoch:")
                        print(cumulative_impact_intervals)

                percentage_been_pruned += self.pruning_step
                print(" >> Pruning progress:", bcolors.BOLD, str(round(percentage_been_pruned * 100, 2)) + "%", bcolors.ENDC)

                model.compile(optimizer=self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
                
                if evaluator is not None and self.test_set is not None:                    
                    robust_preservation = self.robustness_evaluator.evaluate_robustness(model, (test_images, test_labels), self.model_type)

                    # Update score_board and tape_of_moves
                    score_board.append(robust_preservation)
                    print(bcolors.OKGREEN + "[Epoch " + str(epoch_couter) + "]" + str(robust_preservation) + bcolors.ENDC)

                loss, accuracy = self.evaluate()
                accuracy_board.append((round(loss, 4), round(accuracy, 4)))
                    
                tape_of_moves.append(pruned_pairs)
                pruned_pairs = None
            # Check if have pruned enough number of hidden units
            if self.sampler.mode == SamplingMode.BASELINE and percentage_been_pruned >= 0.5:
                print(" >> Maximum pruning percentage has been reached")
                stop_condition = True
            elif not stop_condition and percentage_been_pruned >= self.pruning_target:
                print(" >> Target pruning percentage has been reached")
                stop_condition = True

            # Save the pruned model at each checkpoint or after the last pruning epoch

            self.model = model
            
            if save_file and (epoch_couter % self.EPOCHS_PER_CHECKPOINT == 0 or stop_condition):
                curr_pruned_model_path = pruned_model_path + "_ckpt_" + str(math.ceil(epoch_couter/self.EPOCHS_PER_CHECKPOINT))

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

        tape_filename = "paoding/logs/" + model_name + "-" + timestamp + "-" + str(self.evaluation_batch)
        if evaluator is None:
            tape_filename = tape_filename+"-BENCHMARK"

        if self.sampler.mode == SamplingMode.BASELINE:
            tape_filename += "_tape_baseline.csv"
        else:
            tape_filename = tape_filename + "_tape_" + self.sampler.mode.name + ".csv"

        if os.path.exists(tape_filename):
            os.remove(tape_filename)

        with open(tape_filename, 'w+', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            if evaluator is not None:
                csv_line = [str(eps) for eps in self.target_adv_epsilons]
            else:
                csv_line = []
            csv_line.append('moves,loss,accuracy')
            csv_writer.writerow(csv_line)

            for index, item in enumerate(accuracy_board):
                if evaluator is not None:
                    rob_pres_stat = [item[k] for k in self.target_adv_epsilons]
                else:
                    rob_pres_stat = []
                rob_pres_stat.append(tape_of_moves[index])
                rob_pres_stat.append(accuracy_board[index])
                csv_writer.writerow(rob_pres_stat)
            
            if evaluator is None:
                csv_writer.writerow(["Elapsed time: ", round((end_time - start_time) / 60.0, 3), "minutes /", int(end_time - start_time), "seconds"])

        print("Model pruning accomplished")

    def prune_cnv(self, evaluator=None, save_file=False, pruned_model_path=None, verbose=0):
        if evaluator is not None:
            self.robustness_evaluator = evaluator
            self.target_adv_epsilons = evaluator.epsilons
            self.evaluation_batch = evaluator.batch_size
        test_images, test_labels = self.test_set
        utils.create_dir_if_not_exist("paoding/logs/")
        # utils.create_dir_if_not_exist("paoding/save_figs/")
        
        if save_file and pruned_model_path is None:
            pruned_model_path=self.model_path+"_pruned_conv"

        # Start elapsed time counting
        start_time = time.time()
        pruning_result_dict = self.sampler.nominate_conv(self.model, prune_percentage=self.pruning_target)

        self.model = pruning_result_dict['model']

        # self.model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer= self.optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        loss, accuracy = self.model.evaluate(test_images, test_labels, verbose=2)
        print("Evaluation accomplished -- [ACC]", accuracy, "[LOSS]", loss)   
        
        if evaluator is not None and self.test_set is not None:                    
            robust_preservation = self.robustness_evaluator.evaluate_robustness(self.model, (test_images, test_labels), self.model_type)
        
        if save_file and os.path.exists(pruned_model_path):
            shutil.rmtree(pruned_model_path)
            print("Overwriting existing pruned model ...")

        if save_file:
            self.model.save(pruned_model_path)
            print(" >>> Pruned model saved")
        else:
            print(" >>> Pruned model won't be saved unless you set \"save_file\" True")
            
        # Stop elapsed time counting
        end_time = time.time()
        print("Elapsed time: ", round((end_time - start_time)/60.0, 3), "minutes /", int(end_time - start_time), "seconds")

        print("Pruning accomplished")
 