
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

model_path = 'paoding/models/cifar_10_vgg_19'

(train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()

train_features = training_from_data.resize_img(train_features)
test_features = training_from_data.resize_img(test_features)

print("Training dataset size: ", train_features.shape, train_labels.shape)

optimizer = "RMSprop"
loss_fn ="categorical_crossentropy",

training_from_data.transfer_vgg_19_cifar((train_features, train_labels),
                                        (test_features, test_labels),
                                        model_path,
                                        overwrite=False,
                                        optimizer_config=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=30)

# We need to encode labels into one-hot format
train_labels = tf.keras.utils.to_categorical(train_labels,num_classes=10)
test_labels = tf.keras.utils.to_categorical(test_labels,num_classes=10)

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

model_name = 'VGG19'
target = 0.5
step = 0.025

pruner = Pruner(model_path,
            (test_features, test_labels),
            target=target,
            step=step,
            sample_strategy=sampler,
            model_type=ModelType.OTHER,
            stepwise_cnn_pruning=True)
            #seed_val=42)

pruner.load_model(optimizer=optimizer, loss=loss_fn)

pruner.evaluate(verbose=1)
pruned_model_path = model_path + "_pruned"
pruner.prune(evaluator=None, pruned_model_path=pruned_model_path, model_name=model_name, save_file=True)

pruner.evaluate(verbose=1)
pruner.gc()
