from nnprune.pruner import Pruner
from nnprune.sampler import Sampler
from nnprune.evaluator import Evaluator

def test_pruner():
    pruner = Pruner()
    pruner.load_model("hello")

    sampler = Sampler()
    sampler.nominate_candidate()

    eval = Evaluator()
    eval.evaluate()

    pruner.save_model()

def test_chest_xray_model():
    # Define a list to record each pruning decision
        tape_of_moves = []

        # Define a hash map to store definition intervals for all FC neurons
        big_map = {}

        # Define a list to record benchmark & evaluation per pruning epoch (begins with original model)
        score_board = []
        accuracy_board = []

        epoch_couter = 0
        num_units_pruned = 0

        original_model_path = 'tf_codes/models/chest_xray_cnn'
        pruned_model_path = 'tf_codes/models/chest_xray_cnn_pruned'


        ################################################################
        # Prepare dataset and pre-trained model                        #
        ################################################################
        # The Kaggle chest x-ray dataset contains 2 classes 150x150 (we change to 64x64) color images.

        class_names = ['PNEUMONIA', 'NORMAL']

        data_path = "tf_codes/input/chest_xray"
        (train_images, train_labels), (test_images, test_labels), (
        val_images, val_labels) = training_from_data.load_data_pneumonia(data_path)
        print("Training dataset size: ", train_images.shape, train_labels.shape)

        num_units_first_mlp_layer = 128

        training_from_data.train_pneumonia_binary_classification_cnn((train_images, train_labels),
                                                                    (test_images, test_labels),
                                                                    original_model_path,
                                                                    overwrite=False,
                                                                    epochs=20,
                                                                    data_augmentation=True,
                                                                    val_data=(val_images, val_labels))

        model = tf.keras.models.load_model(original_model_path)
        print(model.summary())
        model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])

        loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
