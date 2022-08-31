
from paoding.pruner import Pruner
from paoding.sampler import Sampler
import os, shutil

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, ReLU, Input, Dropout
import tensorflow as tf
import pandas as pd

import glob as gb 
import numpy as np
from tensorflow.keras import datasets
import paoding.utility.training_from_data as training_from_data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from paoding.utility.option import ModelType, SamplingMode

from sklearn.model_selection import train_test_split

def train_brain_cnn(train_data, test_data, path, overwrite=False,
                            optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5),
                            epochs=30):
    
    # Let's start building a model
    if not os.path.exists(path) or overwrite:
        if os.path.exists(path):
            shutil.rmtree(path)
            print("TRAIN ANYWAY option enabled, create and train a new one ...")
        else:
            print("Model not found, create and train a new one ...")

        filter_size=(5,5)
        model = models.Sequential([
            
            layers.Conv2D(32,kernel_size=filter_size,activation='relu',input_shape=(img_size,img_size,1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(.2),
            
            layers.Conv2D(64,kernel_size=filter_size,activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(.2),
            
            layers.Conv2D(128,kernel_size=filter_size,activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(.2),

            layers.Conv2D(128,kernel_size=filter_size,activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(.2),
            
            layers.Flatten(),
            layers.Dense(512,activation='relu'),
            layers.Dense(512,activation='relu'),            
            layers.Dense(4,activation='softmax')
        ])
        print(model.summary())

        model.compile(optimizer=optimizer_config, loss='sparse_categorical_crossentropy', metrics= ['accuracy'])

        model_es = EarlyStopping(monitor = 'loss',  patience =5 , verbose = 1)
        model_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 6, verbose = 1)

        training_history = model.fit(train_data,
            epochs=epochs,
            callbacks=[model_es, model_rlr],
            validation_data=test_data)

        baseline_results = model.evaluate(test_data, verbose=0)
        test_loss, test_accuracy = baseline_results[0], baseline_results[1]

        #test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)
        print("Final Accuracy achieved is: ", test_accuracy, "with Loss", test_loss)

        model.save(path)
        print("Model has been saved")
        #plt.show()

    else:
        print("Model found, there is no need to re-train the model ...")


# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

locat_training="paoding/experiments/data/brain/Training"
locat_testing = "paoding/experiments/data/brain/Testing"
model_path = 'paoding/experiments/models/brain-cnn'

v=0
for folder in os.listdir(locat_training):
    files=gb.glob(pathname=str(locat_training+"//"+folder+"/*.jpg"))
    x= len(files)
    v=v+x
    print(f"the training images in folder {folder} is {len(files)} ")
print(f"the total images is {v}")

v=0
for folder in os.listdir(locat_testing):
    files=gb.glob(pathname=str(locat_testing+"//"+folder+"/*.jpg"))
    x= len(files)
    v=v+x
    print(f"the testing images in folder {folder} is {len(files)} ")
print(f"the total images is {v}")

# Data augmentation

img_size=80
train_datagen=ImageDataGenerator(rotation_range=5,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1.0/255.0 )

train_generator=train_datagen.flow_from_directory(locat_training,
    class_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    target_size=(img_size,img_size)
)

test_datagen=ImageDataGenerator(rescale=1.0/255.0 )
test_generator=train_datagen.flow_from_directory(
    locat_testing,
    class_mode='binary',
    color_mode='grayscale',
    batch_size=32,
    target_size=(img_size,img_size)
)

train_brain_cnn(train_generator, test_generator, model_path, overwrite=False,
                    optimizer_config = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0001, clipvalue=0.5),
                    epochs=30)

model_name = "M"

sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.STOCHASTIC, params=(0.75, 0.25))

pruner = Pruner(model_path,
        test_generator,
        target=0.1,
        step=0.1,
        sample_strategy=sampler,
        model_type=ModelType.CIFAR,
        seed_val=42)

pruner.load_model(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), 
                    loss = 'sparse_categorical_crossentropy')

pruner.evaluate(verbose=1)

pruner.prune(evaluator=None, model_name=model_name)
pruner.evaluate(verbose=1)