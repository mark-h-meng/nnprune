from nnprune.pruner import Pruner
from nnprune.sampler import Sampler
from nnprune.evaluator import Evaluator

import os
import tensorflow as tf
import nnprune.utility.training_from_data as training_from_data

'''
def test_pruner():
    pruner = Pruner()
    pruner.load_model("hello")

    sampler = Sampler()
    sampler.nominate_candidate()

    eval = Evaluator()
    eval.evaluate()

    pruner.save_model()
'''

def test_chest_xray_model():
    print(os.path.dirname(os.path.realpath(__file__)))
   
    original_model_path = 'nnprune/models/chest_xray_cnn'
    pruned_model_path = 'nnprune/models/chest_xray_cnn_pruned'

    ################################################################
    # Prepare dataset and pre-trained model                        #
    ################################################################
    # The Kaggle chest x-ray dataset contains 2 classes 150x150 (we change to 64x64) color images.
    # Class Names: ['PNEUMONIA', 'NORMAL']
    
    data_path = "nnprune/input/chest_xray"
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

       
    pruner = Pruner(original_model_path, (test_images, test_labels), alpha=0.75)

    pruner.load_model()
    pruner.prune()
    pruner.save_model(pruned_model_path)

test_chest_xray_model()