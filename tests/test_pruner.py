import unittest
from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator

import tensorflow as tf
from tensorflow.keras import datasets
import paoding.utility.training_from_data as training_from_data
from paoding.utility.option import ModelType, SamplingMode


class TestPruning(unittest.TestCase):
    
    def test_00_chest_xray_model_training(self):
        
        original_model_path = 'paoding/models/chest_xray_cnn'

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


    def test_01_kaggle_model_training(self):
        original_model_path = 'paoding/models/kaggle_mlp_3_layer'

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


    def test_02_mnist_model_training(self):
        original_model_path = 'paoding/models/mnist_mlp_5_layer'

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


    def test_03_cifar_10_model_training(self):
        
        original_model_path = 'paoding/models/cifar_10_cnn'
        
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
        
        

    def test_04_cifar_10_model_top_k_training(self):
        original_model_path = 'paoding/models/cifar_10_cnn_k'

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
                                                    optimizer_config=optimizer,
                                                    epochs=30,
                                                    topK=3)
        
    def test_10_chest_xray_model_pruning(self):
        
        original_model_path = 'paoding/models/chest_xray_cnn'
        pruned_model_path = 'paoding/models/chest_xray_cnn_pruned'
        data_path = "paoding/input/chest_xray"
        (train_images, train_labels), (test_images, test_labels), (val_images, val_labels) = training_from_data.load_data_pneumonia(data_path)

        sampler = Sampler()
        sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))   
        evaluator = Evaluator()
        pruner = Pruner(original_model_path, 
                        (test_images, test_labels), 
                        target=0.5,
                        step=0.025,
                        sample_strategy=sampler, 
                        model_type=ModelType.XRAY,
                        seed_val=42)

        pruner.load_model()
        pruner.prune(evaluator=evaluator)
        pruner.save_model(pruned_model_path)

    def test_11_kaggle_model_pruning_wo_eval(self):
        original_model_path = 'paoding/models/kaggle_mlp_3_layer'
        pruned_model_path = 'paoding/models/kaggle_mlp_3_layer_pruned'

        data_path = "paoding/input/kaggle/creditcard.csv"
        (train_features, train_labels), (test_features, test_labels) = training_from_data.load_data_creditcard_from_csv(data_path)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        sampler = Sampler()
        sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

        pruner = Pruner(original_model_path, 
            (test_features, test_labels), 
            target=0.5,
            step=0.025,
            sample_strategy=sampler,  
            input_interval=(-5,5),
            model_type=ModelType.CREDIT,
            seed_val=42)

        pruner.load_model(optimizer)
        pruner.prune(evaluator=None)
        pruner.save_model(pruned_model_path)


    def test_12_mnist_model_pruning(self):
        original_model_path = 'paoding/models/mnist_mlp_5_layer'
        pruned_model_path = 'paoding/models/mnist_mlp_pruned_5_layer'

        (train_features, train_labels), (test_features, test_labels) = datasets.mnist.load_data(path="mnist.npz")
        
        # Normalize pixel values to be between 0 and 1
        train_features, test_features = train_features / 255.0, test_features / 255.0

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        sampler = Sampler()
        sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

        evaluator = Evaluator(epsilons=[0.01, 0.05], batch_size=100)
        pruner = Pruner(original_model_path, 
            (test_features, test_labels), 
            target=0.25,
            step=0.025,
            sample_strategy=sampler, 
            model_type=ModelType.MNIST,
            seed_val=42)

        pruner.load_model(optimizer)
        pruner.prune(evaluator=evaluator)
        pruner.save_model(pruned_model_path)


    def test_13_cifar_10_model_pruning_wo_eval(self):
        original_model_path = 'paoding/models/cifar_10_cnn'
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

    def test_14_cifar_10_model_top_k_pruning(self):
        k = 3
        original_model_path = 'paoding/models/cifar_10_cnn_k'
        pruned_model_path = 'paoding/models/cifar_10_cnn_pruned_k'

        (train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()
        # Normalize pixel values to be between 0 and 1
        train_features, test_features = train_features / 255.0, test_features / 255.0

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        training_from_data.train_cifar_9_layer_cnn((train_features, train_labels),
                                                    (test_features, test_labels),
                                                    original_model_path,
                                                    overwrite=False,
                                                    use_relu=True,
                                                    optimizer_config=optimizer,
                                                    epochs=30,
                                                    topK=3)
        evaluator = Evaluator(k=3,epsilons = [0.01, 0.05], batch_size=50)
        
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
        pruner.prune(evaluator=evaluator)
        pruner.save_model(pruned_model_path)

    if __name__ == '__main__':
        unittest.main()   
