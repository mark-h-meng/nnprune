from enum import Enum

class ModelType(Enum):
    XRAY = 1
    CREDIT = 2
    MNIST = 3
    CIFAR = 4
    OTHER = -1

class SamplingMode(Enum):
    BASELINE = 1
    GREEDY = 2
    STOCHASTIC = 3
    OTHER = -1

class AttackAlogirithm(Enum):
    FGSM = 1
    OTHER = -1
