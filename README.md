# CSE 547: Machine Learning for Big Data
## Author: Brian de silva

This repository contains my homework assignment writeups and code along with some code from my project for CSE 547 Spring 2017.


## Project: Financial fraud detection
### Group: Brian de Silva and Daiwei He

We studied the problem of fraud detection in a synthetic data set (located [here](https://www.kaggle.com/ntnu-testimon/paysim1)). Our goal was to train a model to classify financial transactions as being either legitimate or fradulent. The data set is troubled by some of the same issues that commonly occur in financial data. The cost of misclassifying samples is example-dependent and the ratio of positive and negative examples is highly skewed. We discuss and employ techniques for handling both of these issues. We conclude that the most economical solution is, perhaps unsurprisingly, to combine different approaches for tackling the two problems. However we found that if computational cost is of no concern, one can achieve the best results with a decision tree or random forest modified to take the example-dependent costs into account.

At the moment there are two subdirectories of the Project folder:
1. **Code**: contains some of the Jupyter notebooks written for the project
	* **Costcla.ipynb**: this file contains the bulk of our work on the project. Here we compare the performance of various cost-sensitive classifiers (as implemented in the [CostCla](https://pypi.python.org/pypi/costcla/0.5) package).
	* **data_analysis.ipynb**: here we perform some exploratory data analysis, as the file name suggests. We also examine the effect varying the financial cost of investigating a potentially fraudulent transaction has on the costs of some baseline classifiers.
2. **Report**: contains a pdf of our final report

Note that extra files containing the actual Paysim data (along with the output of some neural networks we trained to classify transactions) are required to run these notebooks.