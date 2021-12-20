# PAODING-DL: A Stochastic Data-free Robustness-preserving Neural Network Pruning Solution

Our python package performs pruning progressively and evaluate robustness automatically. The code is written and tested through Microsoft VS Code.

The execution environment of the experiments is summarised as follows: 
* Tensorflow 2.3.0 (Anaconda, tensorflow-gpu version)
* Python 3.8
* Ubuntu 20.04 LTS

## Instructions

The name of Paoding-DL originates from an ancient Chinese story recorded in book "Zhuang Zi". "Paoding" is the name of a chief with excellent skill in butchering and cooking.

![paoding architecture](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/overall.png)

Our tool is implemented as a trilogy, containing three API components: Pruner, Sampler and Evaluator.

![paoding pruner](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruner.png)

The first component, Pruner, servers as the centre of Paoding-DL. Users import model, perform pruning and evaluation, and then save the pruned model through this components. Four methods plus the initiator function are provided in Pruner class. 

![paoding sampler](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/sampler.png)

The second key element is the Sampler, which manages the sampling strategy (sampling mode and corresponding parameters that to be used to guide pruning) and handles hidden unit pruning operations in an iterative way. Three methods are offered in this class, including one setter, one getter and the method called "nominate" to perform sampling and pruning actions.

![paoding evaluator](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/evaluator.png)

The third component is the Evaluator. The evaluator is used to assess the robustness of the pruned model against adversarial perturbation. Besides the evaluation methods, several setter and getter functions are provided to fine grain the evaluation configuration. The adversarial method supported in current release is the FGSM (Fast gradient sign method), and more types of attack approaches will be implemented and supported in future release.

## Testing and Quick Demo

You may run the test cases given in the package to quicky go through the pruning process. The class "test_prunner" will firstly train a few sample models, then perform pruning in different modes, followed by robustness evaluation for selected pruned models. The entire demonstration may spend more than 10 minutes to finish, depending on the computation power of the machine. Some screenshots below depicts the testing process.

![model training](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/model-training.gif)


![model pruning](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruning-benchmarking-mode.gif)


![model evaluation](https://raw.githubusercontent.com/mark-h-meng/nnprune/main/README/pruning-robustness-assessment.gif)
