
from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
import os
import shutil
import time

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

model_path = 'paoding/models/cifar_10_cnn'

(train_features, train_labels), (test_features,
                                 test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_features, test_features = train_features / 255.0, test_features / 255.0

print("Training dataset size: ", train_features.shape, train_labels.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

round = 0
while(round<1):

    training_from_data.train_cifar_cnn((train_features, train_labels),
                                            (test_features, test_labels),
                                            model_path,
                                            overwrite=False,
                                            use_relu=True,
                                            optimizer_config=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=40)

    sampler = Sampler()
    sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25), recursive_pruning=True)

    model_name = 'CIFAR'
    target = 0.25
    step = 0.05

    evaluator = Evaluator(epsilons=[0.01, 0.05], batch_size=100)
    pruner = Pruner(model_path,
                    (test_features, test_labels),
                    target=target-0.03,
                    step=step,
                    sample_strategy=sampler,
                    model_type=ModelType.CIFAR,
                    stepwise_cnn_pruning=True,
                    seed_val=42)

    pruner.load_model(optimizer=optimizer, loss=loss_fn)

    #pruner.evaluate(verbose=1)
    pruned_model_path = model_path + "_pruned"
    pruner.prune(evaluator=evaluator, pruned_model_path=pruned_model_path,
                model_name=model_name, save_file=True)

    #pruner.evaluate(verbose=1)
    pruner.quantization()
    pruner.gc()
    round += 1
