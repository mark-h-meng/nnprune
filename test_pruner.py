from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator

import os
import tensorflow as tf
from tensorflow.keras import datasets
import paoding.utility.training_from_data as training_from_data
from paoding.utility.option import ModelType, SamplingMode

def test_chest_xray_model():
    
    original_model_path = 'paoding/models/chest_xray_cnn'
    pruned_model_path = 'paoding/models/chest_xray_cnn_pruned'

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The Kaggle chest x-ray dataset contains 2 classes 150x150 (we change to 64x64) color images.
    # Class Names: ['PNEUMONIA', 'NORMAL']
    
    data_path = "paoding/input/chest_xray"
    (train_images, train_labels), (test_images, test_labels), (
    val_images, val_labels) = training_from_data.load_data_pneumonia(data_path)
    print("Training dataset size: ", train_images.shape, train_labels.shape)

    training_from_data.train_pneumonia_binary_classification_cnn((train_images, train_labels),
                                                                    (test_images, test_labels),
                                                                    original_model_path,
                                                                    overwrite=False,
                                                                    epochs=20,
                                                                    data_augmentation=True,
                                                                    val_data=(val_images, val_labels))

    sampler = Sampler(mode=SamplingMode.STOCHASTIC)   
    evaluator = Evaluator()
    pruner = Pruner(original_model_path, 
                    (test_images, test_labels), 
                    sample_strategy=sampler, 
                    alpha=0.75,
                    first_mlp_layer_size=128,
                    model_type=ModelType.XRAY)

    pruner.load_model()
    pruner.prune(evaluator=evaluator)
    pruner.save_model(pruned_model_path)

def test_kaggle_model():
    original_model_path = 'paoding/models/kaggle_mlp_3_layer'
    pruned_model_path = 'paoding/models/kaggle_mlp_3_layer_pruned'

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The MNIST dataset contains 60,000 28x28 greyscale images of 10 digits.
    # There are 50000 training images and 10000 test images.

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

    sampler = Sampler(mode=SamplingMode.STOCHASTIC)   
    evaluator = Evaluator()
    pruner = Pruner(original_model_path, 
        (test_features, test_labels), 
        sample_strategy=sampler, 
        alpha=0.75, 
        input_interval=(-5,5),
        first_mlp_layer_size = 64,
        model_type=ModelType.CREDIT)

    pruner.load_model(optimizer)
    pruner.prune(evaluator=evaluator)
    pruner.save_model(pruned_model_path)


def test_mnist_model():
    original_model_path = 'paoding/models/mnist_mlp_5_layer'
    pruned_model_path = 'paoding/models/mnist_mlp_pruned_5_layer'

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The MNIST dataset contains 60,000 28x28 greyscale images of 10 digits.
    # There are 50000 training images and 10000 test images.

    (train_features, train_labels), (test_features, test_labels) = datasets.mnist.load_data(path="mnist.npz")
    print("Training dataset size: ", train_features.shape, train_labels.shape)
    # Normalize pixel values to be between 0 and 1
    train_features, test_features = train_features / 255.0, test_features / 255.0

    print("Training dataset size: ", train_features.shape, train_labels.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    training_from_data.train_mnist_5_layer_mlp((train_features, train_labels),
                                                   (test_features, test_labels),
                                                   original_model_path,
                                                   overwrite=False,
                                                   use_relu=True,
                                                   optimizer_config=optimizer)

    sampler = Sampler(mode=SamplingMode.STOCHASTIC)   
    evaluator = Evaluator()
    pruner = Pruner(original_model_path, 
        (test_features, test_labels), 
        sample_strategy=sampler, 
        alpha=0.75, 
        first_mlp_layer_size = 128,
        model_type=ModelType.MNIST)

    pruner.load_model(optimizer)
    pruner.prune(evaluator=evaluator)
    pruner.save_model(pruned_model_path)


def test_cifar_10_model():
    original_model_path = 'paoding/models/cifar_10_cnn'
    pruned_model_path = 'paoding/models/cifar_10_cnn_pruned'

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes.
    # There are 50000 training images and 10000 test images.
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


    sampler = Sampler(mode=SamplingMode.STOCHASTIC)   
    evaluator = Evaluator()
    pruner = Pruner(original_model_path, 
        (test_features, test_labels), 
        sample_strategy=sampler, 
        alpha=0.75, 
        first_mlp_layer_size = 128,
        model_type=ModelType.CIFAR)

    pruner.load_model(optimizer)
    pruner.prune(evaluator=evaluator)
    pruner.save_model(pruned_model_path)

print(os.path.dirname(os.path.realpath(__file__)))
   
test_chest_xray_model()
test_kaggle_model()
test_mnist_model()
test_cifar_10_model()