import tensorflow as tf
from tensorflow.keras import datasets

import paoding.utility.training_from_data as training_from_data
from paoding.evaluator import Evaluator
from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.utility.option import ModelType, SamplingMode

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')
model_path = 'paoding/models/mnist_cnn'

################################################################
# Prepare dataset and pre-trained model                        #
################################################################
# The MNIST dataset contains 60,000 28x28 greyscale images of 10 digits.
# There are 50000 training images and 10000 test images.

(train_features, train_labels), (test_features, test_labels) = datasets.mnist.load_data(path="mnist.npz")

# Normalize pixel values to be between 0 and 1
train_features = train_features.reshape(
    train_features.shape[0], 28, 28, 1) / 255.0,
test_features = test_features.reshape(
    test_features.shape[0], 28, 28, 1) / 255.0

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

training_from_data.train_mnist_cnn((train_features, train_labels),
                                   (test_features, test_labels),
                                   model_path,
                                   overwrite=False,
                                   use_relu=True,
                                   optimizer_config=optimizer,
                                   epochs=30)

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.IMPACT, params=(0.75, 0.25))

model_name = 'MNIST'
target = 0.25
step = 0.05

evaluator = Evaluator(epsilons=[0.01, 0.05], batch_size=100)

pruner = Pruner(model_path,
                (test_features, test_labels),
                target=target,
                step=step,
                sample_strategy=sampler,
                model_type=ModelType.MNIST,
                stepwise_cnn_pruning=True,
                seed_val=42)

pruner.load_model(optimizer=optimizer, loss=loss_fn)
pruned_model_path = model_path + "_pruned"
pruner.prune(evaluator=evaluator, pruned_model_path=pruned_model_path,
             model_name=model_name, save_file=True)

pruner.evaluate()

# END OF THE CODE