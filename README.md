# PAODING-DL: A Data-free Robustness-preserving Neural Network Pruning Toolkit

Our python package performs pruning progressively and evaluate robustness automatically. The code is written and tested through Microsoft VS Code on Linux Ubuntu OS.

The execution environment of the experiments is summarised as follows: 
* Tensorflow 2.3.0 (Anaconda, tensorflow-gpu version) and above
* Python 3.8 and above

## Instructions

The name of Paoding-DL originates from an ancient Taoism story recorded in book "Zhuang Zi" that was published in 3rd century BC, under the title "Essentials for Nurturing Life" of the "Inner Chapters". "Paoding" is the name of a chief with excellent skill in butchering and cooking.

![paoding architecture](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/overall.png)

Our tool is implemented as a trilogy, containing three API components: Pruner, Sampler and Evaluator.

![paoding pruner](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruner.png)

The first component, Pruner, servers as the centre of Paoding-DL. Users import model, perform pruning and evaluation, and then save the pruned model through this components. Four methods plus the initiator function are provided in Pruner class. 

![paoding sampler](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/sampler.png)

The second key element is the Sampler, which manages the sampling strategy (sampling mode and corresponding parameters that to be used to guide pruning) and handles hidden unit pruning operations in an iterative way. Three methods are offered in this class, including one setter, one getter and the method called "nominate" to perform sampling and pruning actions.

![paoding evaluator](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/evaluator.png)

The third component is the Evaluator. The evaluator is used to assess the robustness of the pruned model against adversarial perturbation. Besides the evaluation methods, several setter and getter functions are provided to fine grain the evaluation configuration. The adversarial method supported in current release is the FGSM (Fast gradient sign method), and more types of attack approaches will be implemented and supported in future release.

## Testing and Quick Demo

To use Paoding-dl, all you need is to import the three key components in your python code.

```python
from paoding.pruner import Pruner
from paoding.sampler import Sampler
from paoding.evaluator import Evaluator
```

First you need to provide the path of a pretrained TensorFlow model, otherwise you may make use of paoding's API to train a (pretty standard) model like the code snippets below:

```python
# Specify the path of the model
model_path = 'pretrain_model'

(train_features, train_labels), (test_features,
                                 test_labels) = datasets.mnist.load_data(path="mnist.npz")

# Normalize pixel values to be between 0 and 1
train_features = train_features.reshape(train_features.shape[0], 28, 28, 1) / 255.0,
test_features = test_features.reshape(test_features.shape[0], 28, 28, 1) / 255.0

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

training_from_data.train_mnist_cnn((train_features, train_labels), 
        (test_features, test_labels), 
        model_path,
        optimizer_config=optimizer,
        epochs=30)
```

Next, let's initialize the sampler with a sampling strategy, and define the rule of pruning in Pruner. If the pretrained model is trained with MNIST or CIFAR-10, you may also try our built-in FGSM adversarial evaluation, which is defined in Evaluator.

```python
sampler = Sampler()
sampler.set_strategy(mode=SamplingMode.IMPACT, params=(0.5, 0.5))

# set evaluator None if you do not want to perform adversarial evaluation
evaluator = Evaluator(epsilons=[0.01, 0.05], batch_size=100)

pruner = Pruner(model_path,
            (test_features, test_labels),
            target=0.25,
            step=0.05,
            sample_strategy=sampler,
            model_type=ModelType.MNIST,
            stepwise_cnn_pruning=True)

# load the pretrained model according to the model path provided while initializing the pruner
pruner.load_model(optimizer=optimizer, loss=loss_fn)

pruned_model_path = model_path + "_pruned"

# perform pruning, and you will be able to find the pruned model saved at the designated path
pruner.prune(evaluator=evaluator, pruned_model_path=pruned_model_path, model_name='MNIST', save_file=True)
```

You may also run the test cases given in the package to quicky go through the pruning process. The class "test_prunner" will firstly train a few sample models, then perform pruning in different modes, followed by robustness evaluation for selected pruned models. The entire demonstration may spend a few minutes to finish, depending on the computation power of the machine. Some screenshots below depicts the testing process.

![model training](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/model-training.gif)


![model pruning](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruning-benchmarking-mode.gif)


![model evaluation](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruning-robustness-assessment.gif)

## Acknowledgment

The implementation of convolutional layer pruning is inspired by Keras-surgeon under MIT License (https://github.com/BenWhetton/keras-surgeon).

The author would also thank GitHub and PyPI for the free hosting service provided.