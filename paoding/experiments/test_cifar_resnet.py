import os, re, time, json
import numpy as np
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

model_path = 'paoding/models/cifar_10_resnet_50'

(train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()

train_X = training_from_data.preprocess_image_input_resnet(train_features)
test_X = training_from_data.preprocess_image_input_resnet(test_features)

print("Training dataset size: ", train_features.shape, train_labels.shape)

optimizer_config = "SGD"
loss_fn ="sparse_categorical_crossentropy"

training_from_data.transfer_resnet_50((train_X, train_labels),
                                        (test_X, test_labels), 
                                        model_path, overwrite=False, 
                                        optimizer_config = optimizer_config,
                                        loss_fn = loss_fn,
                                        epochs=3)

# We need to encode labels into one-hot format
#train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=10)
#test_labels = tf.keras.utils.to_categorical(test_labels,num_classes=10)

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

model_name = 'RESNET'
target = 0.05
step = 0.025

pruner = Pruner(model_path,
            (test_X, test_labels),
            target=target,
            step=step,
            sample_strategy=sampler,
            model_type=ModelType.CIFAR,
            stepwise_cnn_pruning=True,
            surgery_mode=False,
            seed_val=42)

pruner.load_model(optimizer=optimizer_config, loss=loss_fn)

pruner.evaluate(verbose=1, batch_size=64)
pruned_model_path = model_path + "_pruned"
pruner.prune(evaluator=None, pruned_model_path=pruned_model_path, model_name=model_name, save_file=True)

pruner.evaluate(verbose=1, batch_size=64)
pruner.gc()
