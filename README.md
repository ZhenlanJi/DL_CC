# DL_CC
This repository belongs to our submitted manuscript:
> CC: Causality-Aware Coverage Criterion for Deep Neural Networks
You can find the code for our paper in this repository.

## Introduction
Deep neural network (DNN) testing approaches have grown fast in recent years to test the correctness and robustness of DNNs. In particular, DNN coverage criteria are frequently used to evaluate the quality of a test suite, and a number of coverage criteria based on neuron-wise, layer-wise, and path-trace-wise coverage patterns have been published to date. However, we see that existing criteria are insufficient to represent how one neuron would influence subsequent neurons; hence, we lack a concept of how neurons, when functioning as causes and effects, might jointly make a DNN prediction.

On the basis of recent advances in interpreting DNN internals using causal inference, we present the first causality-aware DNN coverage criterion, which evaluates a test suite by quantifying the extent to which the suite provides new causal relations for testing DNNs. Performing standard causal inference on DNNs presents both theoretical and practical hurdles. We introduce CC (causal coverage), a practical and efficient coverage criterion that integrates a number of optimizations using DNN domain-specific knowledge. We illustrate the efficacy of CC utilizing both diverse, real-world test inputs and adversarial inputs, such as adversarial examples (AEs) and backdoor inputs. We demonstrate that CC outperforms previous DNN criteria under various settings with moderate cost.


## Dependency

```
python 3.9
torch
torchvision
pyflann
scipy
sklearn
numpy
causallearn
torchattacks
```

## Usage

```bash
# default run our rq1 experiment of CC on CIFAR10
python run.py

# use following command to see more args
python run.py --help
```

## Notice

1. ./backdoor_coverage.svg is the coverage pattern visualization of CC on CIFAR10 with backdoor inputs for RQ4.
2. We use several open-source code in our project, including [DeepGauge](https://github.com/hfeniser/DeepSmartFuzzer),
[SA](https://github.com/coinse/sadl) and [backdoor](https://github.com/csdongxian/ANP_backdoor)
1. We add 5 figures in ./coverage_fig to illustrate how each coverage criteria works.