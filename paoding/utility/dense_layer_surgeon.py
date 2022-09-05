#!/usr/bin/python3
__author__ = "Mark H. Meng"
__copyright__ = "Copyright 2022, National University of S'pore and A*STAR"
__credits__ = ["G. Bai", "H. Guo", "S. G. Teo", "J. S. Dong"]
__license__ = "MIT"


import tensorflow as tf
import numpy as np
import os, shutil

def load_param_and_config(model, debugging_output=False):
    # Take a 3 layer MLP as example:
    # "w" is a list, where "w[0]" is empty as it specifies the input layer
    #  "w[1]" contains paramters for the 1st hidden layer, where
    #   "w[1][0] is in shape of 784x128" and w[1][1] is in shape of 128x1 (bias);
    #  "w[2]" contains paramters for the 2nd hidden layer, where
    #   "w[2][0] is in shape of 128x10" and w[2][1] is in shape of 10x1 (bias);

    g = []
    w = []
    layer_index = 0
    for layer in model.layers:
        g.append(layer.get_config())
        w.append(layer.get_weights())
        if "dense" in layer.name:
            num_units = g[layer_index]['units']
            num_prev_neurons = len(w[layer_index][0])
            if debugging_output:
                print("TYPE OF ACTIVATION: ", g[layer_index]['activation'])
                print("CURRENT LAYER: ", g[layer_index]['name'])
                print("NUM OF UNITS: ", num_units)
                print("NUM OF CONNECTION PER UNIT: ", num_prev_neurons)
        layer_index += 1
    return (w, g)


def trim_weights(model, pruned_pairs):
    (w, g) = load_param_and_config(model)
    cut_list_entire_model = [] # Add a zero for the first layer (usually a Flatten layer)

    for layer_idx, pairs_at_layer in enumerate(pruned_pairs):
        if len(pairs_at_layer) == 0:
            cut_list_entire_model.append(0)
        else:
            cut_list_curr_layer = []
            for (node_a, node_b) in pairs_at_layer:
                cut_list_curr_layer.append(node_b)
            cut_list_curr_layer.sort()
            cut_list_entire_model.append(len(cut_list_curr_layer))

            for node in cut_list_curr_layer:
                # Now let's remove the "node_b" hidden unit at the current layer
                ## Cut the connections in the current layer
                list_new_w_layer_idx_0 = []
                for index, prev_layer_unit in enumerate(w[layer_idx][0]):
                    assert node < len(
                        prev_layer_unit), "The index of hidden unit to cut is larger than the size of curr layer"
                    list_new_w_layer_idx_0.append(np.delete(prev_layer_unit, node, 0))

                assert len(w[layer_idx][0]) == len(list_new_w_layer_idx_0), \
                    "The length of original param at layer " + layer_idx + " should be equal to the newly appened one"

                w[layer_idx][0] = np.array(list_new_w_layer_idx_0)
                ## Cut the bias in the current layer
                w[layer_idx][1] = np.delete(w[layer_idx][1], node, 0)
                ## Cut the connections in the next layer
                w[layer_idx + 1][0] = np.delete(w[layer_idx + 1][0], node, 0)

    cut_list_entire_model.append(0) # Add a zero for the output layer (usually a Dense layer)
    return w, g, cut_list_entire_model


def create_pruned_model(original_model, pruned_list, test_set, path):
    # Let's start building a model
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print("The given path is not available, overwriting the old file ...")

    new_weights, config, cut_list = trim_weights(original_model, pruned_list)

    (x_test, y_test) = test_set

    pruned_model = tf.keras.models.Sequential()

    for layer_idx, layer_config in enumerate(config):
        print("Constructing layer", layer_idx)
        if layer_idx == 0:
            if 'flatten' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Flatten(input_shape=config[0]['batch_input_shape'][1:]))
            elif 'conv2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Conv2D(config[layer_idx]['filters'],
                                                        kernel_size=config[layer_idx]['kernel_size'],
                                                        activation=config[layer_idx]['activation'],
                                                        input_shape=config[0]['batch_input_shape'][1:],
                                                        padding=config[0]['padding'],
                                                        strides=config[0]['strides'],
                                                        trainable=False))
        else:
            if 'dense' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Dense(config[layer_idx]['units'] - cut_list[layer_idx],
                                                   activation=config[layer_idx]['activation'],
                                                   trainable=False))
            elif 'conv2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Conv2D(config[layer_idx]['filters'],
                                                    kernel_size=config[layer_idx]['kernel_size'],
                                                    activation=config[layer_idx]['activation'],
                                                    padding=config[layer_idx]['padding'],
                                                    strides=config[layer_idx]['strides'],
                                                    trainable=False))
            elif 'flatten' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.Flatten())
            elif 'max_pooling2d' in config[layer_idx]['name']:
                pruned_model.add(tf.keras.layers.MaxPooling2D(pool_size=config[layer_idx]['pool_size'],
                                                        strides=config[layer_idx]['strides'],
                                                        trainable=False))
            else:
                print("Unable to construct layer", layer_idx, "due to incompatible layer type")

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for index, layer in enumerate(pruned_model.layers):
        layer.set_weights(new_weights[index])

    pruned_model.compile(optimizer='adam',
                         loss=loss_fn,
                         metrics=['accuracy'])
    print(pruned_model.summary())
    loss, accuracy = pruned_model.evaluate(x_test, y_test)
    print(loss, accuracy)
    pruned_model.save(path)
