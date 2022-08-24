
from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator

import tensorflow as tf
from tensorflow.keras import datasets
import paoding.utility.training_from_data as training_from_data
from paoding.utility.option import ModelType, SamplingMode

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
original_model_path = 'paoding/models/cifar_10_cnn'

(train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_features, test_features = train_features / 255.0, test_features / 255.0

print("Training dataset size: ", train_features.shape, train_labels.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

training_from_data.train_cifar_9_layer_cnn((train_features, train_labels),
                                            (test_features, test_labels),
                                            original_model_path,
                                            overwrite=False,
                                            use_relu=True,
                                            optimizer_config=optimizer)

pruned_model_path = 'paoding/models/cifar_10_cnn_pruned'

(train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_features, test_features = train_features / 255.0, test_features / 255.0

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

training_from_data.train_cifar_9_layer_cnn((train_features, train_labels),
                                            (test_features, test_labels),
                                            original_model_path,
                                            overwrite=False,
                                            use_relu=True,
                                            optimizer_config=optimizer)

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

pruner = Pruner(original_model_path, 
    (test_features, test_labels), 
    target=0.25,
    step=0.025,
    sample_strategy=sampler, 
    model_type=ModelType.CIFAR,
    seed_val=42)

pruner.load_model(optimizer)
pruner.prune(evaluator=None)
pruner.save_model(pruned_model_path)