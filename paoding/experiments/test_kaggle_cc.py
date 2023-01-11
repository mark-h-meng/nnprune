
from paoding.pruner import Pruner
from paoding.sampler import Sampler
import os, shutil, time

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ReLU, Input, Dropout
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import datasets
import paoding.utility.training_from_data as training_from_data
from paoding.utility.option import ModelType, SamplingMode

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
original_model_path = 'paoding/models/kaggle_mlp_3_layer'
data_path = "paoding/input/kaggle/creditcard.csv"
(train_features, train_labels), (test_features, test_labels) = training_from_data.load_data_creditcard_from_csv(data_path)
print("Training dataset size: ", train_features.shape, train_labels.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Train a 3 layer FC network: 28 * 64 (ReLU) * 64 (ReLU) * 1 (Sigmoid)
training_from_data.train_creditcard_3_layer_mlp((train_features, train_labels),
                                            (test_features, test_labels),
                                            original_model_path,
                                            overwrite=False,
                                            optimizer_config=optimizer)
LARGE_MODEL = True
model_name = "CREDIT"
pruned_model_path = 'paoding/models/kaggle_mlp_3_layer_pruned'

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25), recursive_pruning=False)

repeat = 1
round = 0

while(round<repeat):

    pruner = Pruner(original_model_path, 
            (test_features, test_labels), 
            target=0.5,
            step=0.025,
            sample_strategy=sampler,  
            input_interval=(-5,5),
            model_type=ModelType.CREDIT,
            seed_val=42,
            surgery_mode=False,
            batch_size=64)

    pruner.load_model(optimizer, loss=tf.keras.losses.BinaryCrossentropy())
    pruner.prune(evaluator=None, pruned_model_path=pruned_model_path, model_name=model_name, save_file=True)
    
    #if round == repeat-1:
    #    pruner.quantization()
    #pruner.gc()

    round += 1