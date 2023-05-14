# ML-Algorithms

## Summary

This repository contains a collection of machine learning models learnt through my time as an NUS student as well as what I have picked up on my own through my own self-learning.

## Description

The source code in this repository is meant to provide a high-level demonstration of the implementation of the ML models I have learnt and understood. They do not contain the mathematical proofs behind this models.

The models are implemented using scikit-learn, a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.

The datasets used for the demonstration of these models are from the toy datasets provided in the scikit-learn library.

### Machine Learning Related Modules taken by me so far in NUS:

1. EE2211 Introduction to Machine Learning
2. CS3244 Machine Learning (Planning to take)

## How to run

### Virtual Environment

For this repository, I chose to use Virtualenv to create a virtual environment to run the source code enclosed in the directory. You may choose to use other packages/libraries to implement a virtual environment e.g. (Conda).

To install Virtualenv:

```bash
pip install virtualenv
```

To enable and activate the virtual environment:

```bash
cd ML-Algorithms
virtualenv venv
```

Subsequently, this creats a `venv/` directory in your project where all dependencies are installed. You need to activate it first though (in every terminal instance where you are working on your project):

```bash
source venv/bin/activate
```

### Running Models

To run the models, first install the required packages in the `requirements.txt` file.

To do so, run the following command:

```bash
pip install -r requirements.txt
```

Subsequently to run the model, use the following command:

```bash
python3 /path/to/file/filename.py
```

where filename is the name of the file. All files can be found in the algorithms directory and are seperated by the type of model (e.g. Classification, Clustering, Model Selection, Regression).
