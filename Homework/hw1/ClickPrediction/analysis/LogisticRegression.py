import math
import matplotlib.pyplot as plt

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from analysis.BasicAnalysis import BasicAnalysis

# This class represents the weights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    self.w_tokens = {}
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}
    
  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string
  
  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)
  
  # @return {Int} the l0 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:
  # ==========================
  # Helper function to compute inner product w_0 + w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    # First few components of inner product
    output =  1.0 * (weights.w0 + weights.w_age * instance.age + weights.w_depth * instance.depth
      + weights.w_position * instance.position + weights.w_gender * instance.gender)
    # Components of inner product involving tokens
    for token in instance.tokens:
      if (token in weights.w_tokens):
        output += weights.w_tokens[token]

    return output
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, tokens, weights, now, step, lambduh):
    
    for token in tokens:
      if (token in weights.w_tokens):
        before = weights.access_time[token]
        weights.w_tokens[token] *= pow(1. - step * lambduh, now - before + 1)
      else:
        weights.access_time[token] = now

    return

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    weights = Weights()

    # Number of steps to perform between each update
    output_steps = 100

    # Variable to record the running loss
    loss = 0.0

    # For each data point:
    while dataset.hasNext():
      instance = dataset.nextInstance()

      # Perform delayed regularization
      self.perform_delayed_regularization(instance.tokens,weights,dataset.counter,step,lambduh)

      # Predict the label, record the loss
      ip = self.compute_weight_feature_product(weights,instance)
      if ip > 0:
        prediction = 1.
      else:
        prediction = 0.

      loss += pow(prediction - instance.clicked, 2)


      if ((dataset.counter % output_steps)==0) or (not dataset.hasNext()):
        if dataset.hasNext():
          # (predictions,loss) = self.predict(weights,dataset)
          avg_loss[(dataset.counter / output_steps) - 1] = loss / dataset.counter
          # print "Average loss after %d iterations: %f" %(dataset.counter, loss / dataset.counter)
        else:
          # (predictions,loss) = self.predict(weights,dataset)
          avg_loss[dataset.counter / output_steps] = loss / dataset.counter
          # print "Average loss after %d iterations: %f" %(dataset.counter, loss / dataset.counter)

        
      # Compute w0 + <w, x>, and gradient
      exp_inner_product = math.exp(ip)
      coeff = instance.clicked - exp_inner_product / (1 + exp_inner_product)
      
      # Update weights along the negative gradient
      weights.w_age += step * instance.age * coeff
      weights.w_gender += step * instance.gender * coeff
      weights.w_depth += step * instance.depth * coeff
      weights.w_position += step * instance.position * coeff
      weights.w0 += step * coeff

      # Update weights corresponding to tokens
      for token in instance.tokens:
        if token in weights.w_tokens:
          weights.w_tokens[token] += step * coeff
        else:
          weights.w_tokens[token] =  step * coeff

      
    dataset.reset()
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    # exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
    # return exp_inner_product / (1 + exp_inner_product)


    predictions = [None] * dataset.size
    count = dataset.counter             # Record position in dataset so it can be returned the same way

    # # Check if we're working with training or testing data
    # if dataset.has_label:
    #   # Get predictions for the later parts of dataset
    #   while dataset.hasNext():
    #     instance = dataset.nextInstance()
    #     exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
    #     predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)
    #     loss += pow(predictions[dataset.counter-1] - instance.clicked,2)

    #   # Get predictions for the earlier part of the dataset
    #   dataset.reset()
    #   while dataset.counter < count:
    #     instance = dataset.nextInstance()
    #     exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
    #     predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)
    #     loss += pow(predictions[dataset.counter-1] - instance.clicked,2)

    # else:


    # Get predictions for the later parts of dataset
    while dataset.hasNext():
      instance = dataset.nextInstance()
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
      predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)

    # Get predictions for the earlier part of the dataset
    dataset.reset()
    while dataset.counter < count:
      instance = dataset.nextInstance()
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
      predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)


    return predictions
  
  
if __name__ == '__main__':
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552

  # TRAININGSIZE = 10001
  # TESTINGSIZE = 10001

  training = DataSet("../../data/train.txt", True, TRAININGSIZE)
  testing = DataSet("../../data/test.txt", False, TESTINGSIZE)

  # Hyperparameters:
  # lambduh = 0                                 # Regularization parameter
  # step = [1.e-3, 1.e-2, 5.e-2]                # SGD stepsizes

  lambduh = [0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014];
  step = 0.05

  # avg_loss = [[None] * (TRAININGSIZE / 100 + 1)] * (len(step)*len(lambduh)          # 2D array for storing avg loss
  avg_loss = [None] * (TRAININGSIZE / 100 + 1)
  weight_vec = [None] * len(lambduh)
  test_rmse_vec = [None] * len(lambduh)

  # Set up some objects we will need
  l = LogisticRegression()
  evaluator = EvalUtil()

  # # Get average CTR for test set for later use
  # analyze = BasicAnalysis()
  # test_avg_ctr = analyze.average_ctr(testing)
  # print "Computing average CTR for test set..."
  test_avg_ctr = 0.033655


  # Train the logistic regression model for various stepsizes
  for k in range(len(lambduh)):
    print "Training Logistic Regression..."
    weights = l.train(training,lambduh[k],step,avg_loss)
    print "Done.\nAverage loss on the training set (eta,lambda) = (%f,%f): %f" %(step,lambduh[k], avg_loss[k])
    weight_vec[k] = weights.l2_norm()
    print "l2 norm of weights (eta,lambda) = (%f,%f): %f" %(step,lambduh[k], weights.l2_norm())

    # Get the RMSE on the test data using logistic regression
    
    # (test_predictions, dummy) = l.predict(weights,testing)
    test_predictions = l.predict(weights,testing)
    # count = 0
    # while testing.hasNext():
    #   instance = testing.nextInstance()
    #   test_predictions[count] = l.predict(weights, instance)
    #   count += 1

    test_rmse = evaluator.eval("../../data/test_label.txt",test_predictions)
    test_rmse_vec[k] = test_rmse
    print "RMSE on the test set using logistic regression (eta,lambda) = (%f,%f): %f" %(step,lambduh[k], test_rmse)

    # # Get RMSE on the test data using baseline
    # test_avg_rmse = evaluator.eval_baseline("../../data/test_label.txt",test_avg_ctr)
    # print "RMSE on the tes set using average CTR (eta,lambda) = (%f,%f): %f" %(step[k],lambduh[k], test_avg_rmse)


  # # Plot the error as a function of iterations for different eta values
  # its = range(100,TRAININGSIZE,100)
  # its.append(TRAININGSIZE)
  # plt.plot(its,avg_loss[0],its,avg_loss[1],its,avg_loss[2])
  # plt.legend(['eta=0.001','eta=0.01','eta=0.05'])
  # plt.xlabel("Steps")
  # plt.ylabel("Average loss")
  # plt.title("Average Loss During Logistic Regression Training")
  # plt.show()

  # Plot the l2 norms of weights as a function of lambda
  plt.plot(lambduh,weight_vec,'b-o')
  plt.xlabel('Lambda')
  plt.ylabel('l2 norm of weights')
  plt.title('l2 norm of weights as regularization is increased')
  plt.legend(range())

  # Plot the RMSE as a function of lambda
  plt.plot(lambduh,rmse_vec,'b-o')
  plt.xlabel('Lambda')
  plt.ylabel('RMSE')
  plt.title('RMSE for predicted CTR on test set as regularization is increased')
  plt.legend(range())
