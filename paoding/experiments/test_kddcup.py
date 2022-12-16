
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

# Convert numeric variables to Z scores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

# Dummy encoding for categorical variables ([1,0],[0,1],[0,0] for 'Red', 'Green' and 'Blue')
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


def train_kdd99_5_layer_mlp(train_data, test_data, path, overwrite=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=30):
    BATCH_SIZE = 512

    (x, y)=train_data
    (test_features, test_labels)=test_data
    train_features, val_features, train_labels, val_labels = train_test_split(x, y,
                                                                              test_size=0.2, train_size=0.8)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(train_features.shape[-1],)))
        #model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        #model.add(layers.Dropout(0.5))
        model.add(layers.Dense(23, activation='softmax'))

        print(model.summary())

        # model.compile(optimizer=optimizer_config, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc'])
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        training_history = model.fit(train_features, train_labels,
                                     batch_size=BATCH_SIZE,
                                     epochs=epochs,
                                     callbacks=[early_stopping],
                                     validation_data=(val_features, val_labels))

        baseline_results = model.evaluate(test_features, test_labels, verbose=0)
        test_loss, test_accuracy = baseline_results[0], baseline_results[1]

        #test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
        print("Final Accuracy achieved is: ", test_accuracy, "with Loss", test_loss)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        #plt.show()

    else:
        print("Model found, there is no need to re-train the model ...")

def train_kdd99_7_layer_mlp(train_data, test_data, path, overwrite=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                            epochs=30):
    BATCH_SIZE = 512

    (x, y)=train_data
    (test_features, test_labels)=test_data
    train_features, val_features, train_labels, val_labels = train_test_split(x, y,
                                                                              test_size=0.2, train_size=0.8)

    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True)

        model = models.Sequential()
        model.add(layers.Dense(512, activation='relu', input_shape=(train_features.shape[-1],)))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(23, activation='softmax'))

        print(model.summary())

        # model.compile(optimizer=optimizer_config, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc'])
        model.compile(optimizer=optimizer_config, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

        training_history = model.fit(train_features, train_labels,
                                     batch_size=BATCH_SIZE,
                                     epochs=epochs,
                                     callbacks=[early_stopping],
                                     validation_data=(val_features, val_labels))

        baseline_results = model.evaluate(test_features, test_labels, verbose=0)
        test_loss, test_accuracy = baseline_results[0], baseline_results[1]

        #test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
        print("Final Accuracy achieved is: ", test_accuracy, "with Loss", test_loss)

        model.save(path)
        dot_img_file = path + '.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
        print("Model has been saved")
        #plt.show()

    else:
        print("Model found, there is no need to re-train the model ...")


# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

# The original dataset can be found at https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data
data_path = 'paoding/experiments/data/kdd-cup/kddcup.data_10_percent_corrected'
model_path = 'paoding/experiments/models/kddcup-fc'

# Load 494021 data entries, including 41 columns
data = pd.read_csv(data_path, header=None)

# Full description please refer to http://kdd.ics.uci.edu/databases/kddcup99/task.html
data.columns = [
    'duration', # length (number of seconds) of the connection, range [0, 58329]
    'protocol_type', # TCP, UDP, ICMP
    'service', # Service type of the target machine, 70 types 'http_443', 'http_8001', 'imap4', etc
    'flag', # Status of connection in 11 types such as 'S0', 'S1' and 'S2', etc.
    'src_bytes', # range [0,1379963888]
    'dst_bytes', # range [0.1309937401]
    'land', # 1 if the connection is from/to the same machine/port, otherwise 0
    'wrong_fragment', # range [0,3]
    'urgent', # Num of urgent packet, range [0,14]
    'hot', # Time of access of sensitive dir, range [0,101]
    'num_failed_logins', #  range [0,5]
    'logged_in', # 1 if successful logon, otherwise 0
    'num_compromised', # range [0,7479]
    'root_shell', # 1 if obtained root shell, otherwise 0
    'su_attempted', # 1 if contains "su root" command, otherwise 0
    'num_root', #  Num of times of root access, range [0,7468]
    'num_file_creations', # range [0,100]
    'num_shells', # range [0,5]
    'num_access_files', # range [0,9]
    'num_outbound_cmds', # always 0
    'is_host_login', # 1 if login user is host, otherwise 0
    'is_guest_login', # 1 if login user is guest, otherwise 0
    'count', # range [0,511]
    'srv_count', # range [0,511]
    'serror_rate', # range [0.00,1.00]
    'srv_serror_rate', # range [0.00,1.00]
    'rerror_rate', # range [0.00,1.00]
    'srv_rerror_rate', # range [0.00,1.00]
    'same_srv_rate', # range [0.00,1.00]
    'diff_srv_rate', # range [0.00,1.00]
    'srv_diff_host_rate', # range [0.00,1.00]
    'dst_host_count', # range [0,255]
    'dst_host_srv_count', # range [0,255]
    'dst_host_same_srv_rate', # range [0.00,1.00]
    'dst_host_diff_srv_rate', # range [0.00,1.00]
    'dst_host_same_src_port_rate', # range [0.00,1.00]
    'dst_host_srv_diff_host_rate', # range [0.00,1.00]
    'dst_host_serror_rate', # range [0.00,1.00]
    'dst_host_srv_serror_rate', # range [0.00,1.00]
    'dst_host_rerror_rate', # range [0.00,1.00]
    'dst_host_srv_rerror_rate', # range [0.00,1.00]
    'outcome' # Tag
]

# Preprocess each variable accordingly
encode_numeric_zscore(data, 'duration')
encode_text_dummy(data, 'protocol_type')
encode_text_dummy(data, 'service')
encode_text_dummy(data, 'flag')
encode_numeric_zscore(data, 'src_bytes')
encode_numeric_zscore(data, 'dst_bytes')
encode_text_dummy(data, 'land')
encode_numeric_zscore(data, 'wrong_fragment')
encode_numeric_zscore(data, 'urgent')
encode_numeric_zscore(data, 'hot')
encode_numeric_zscore(data, 'num_failed_logins')
encode_text_dummy(data, 'logged_in')
encode_numeric_zscore(data, 'num_compromised')
encode_numeric_zscore(data, 'root_shell')
encode_numeric_zscore(data, 'su_attempted')
encode_numeric_zscore(data, 'num_root')
encode_numeric_zscore(data, 'num_file_creations')
encode_numeric_zscore(data, 'num_shells')
encode_numeric_zscore(data, 'num_access_files')
encode_numeric_zscore(data, 'num_outbound_cmds')
encode_text_dummy(data, 'is_host_login')
encode_text_dummy(data, 'is_guest_login')
encode_numeric_zscore(data, 'count')
encode_numeric_zscore(data, 'srv_count')
encode_numeric_zscore(data, 'serror_rate')
encode_numeric_zscore(data, 'srv_serror_rate')
encode_numeric_zscore(data, 'rerror_rate')
encode_numeric_zscore(data, 'srv_rerror_rate')
encode_numeric_zscore(data, 'same_srv_rate')
encode_numeric_zscore(data, 'diff_srv_rate')
encode_numeric_zscore(data, 'srv_diff_host_rate')
encode_numeric_zscore(data, 'dst_host_count')
encode_numeric_zscore(data, 'dst_host_srv_count')
encode_numeric_zscore(data, 'dst_host_same_srv_rate')
encode_numeric_zscore(data, 'dst_host_diff_srv_rate')
encode_numeric_zscore(data, 'dst_host_same_src_port_rate')
encode_numeric_zscore(data, 'dst_host_srv_diff_host_rate')
encode_numeric_zscore(data, 'dst_host_serror_rate')
encode_numeric_zscore(data, 'dst_host_srv_serror_rate')
encode_numeric_zscore(data, 'dst_host_rerror_rate')
encode_numeric_zscore(data, 'dst_host_srv_rerror_rate')

LARGE_MODEL = True
model_name = "KDDCUP"

# Drop the "num_outbound_cmds" column because all data has its value equal 0
data.dropna(inplace=True, axis=1)

# The first 41 columns are used as "x", and the last column is "y"
x_columns = data.columns.drop('outcome')
X = data[x_columns].values
dummies = pd.get_dummies(data['outcome'])
outcomes = dummies.columns
num_classes = len(outcomes)
Y = dummies.values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=42)

train_features, val_features, train_labels, val_labels = train_test_split(x_train, y_train,
                                                                              test_size=0.2, train_size=0.8)

repeat = 1
test_modes=[SamplingMode.BASELINE, SamplingMode.BASELINE, SamplingMode.IMPACT]
recursive_modes=[False, True, True]

for index, prune_mode in enumerate(test_modes):
    round = 0
    if prune_mode==SamplingMode.BASELINE:
        total_runs = 1 
    else:
        total_runs = repeat

    while(round < total_runs):

        if not LARGE_MODEL:
            train_kdd99_5_layer_mlp((x_train, y_train), (x_test, y_test), model_path, overwrite=False,
                                    optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                                    epochs=10)
        else:
            train_kdd99_7_layer_mlp((x_train, y_train), (x_test, y_test), model_path, overwrite=False,
                                    optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001),
                                    epochs=10)

        sampler = Sampler()
        sampler.set_strategy(mode=prune_mode, recursive_pruning=recursive_modes[index])

        if recursive_modes[index]:
            target = 0.75
        else:
            target = 0.5
        step = 0.03125


        pruner = Pruner(model_path,
                    (x_test, y_test),
                    target=target,
                    step=step,
                    sample_strategy=sampler,
                    model_type=ModelType.KDD,
                    stepwise_cnn_pruning=True,
                    #seed_val=42,
                    surgery_mode=True)

        pruner.load_model(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True))

        pruner.prune(evaluator=None, pruned_model_path=model_path+"_pruned", model_name=model_name, save_file=True)

        pruner.gc()

        round += 1
