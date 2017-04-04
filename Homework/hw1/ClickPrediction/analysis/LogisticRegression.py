import math

from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

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
    output =  1.0 * (weights.w0 + weights.age * instance.age + weights.depth * instance.depth
      + weights.position * instance * position + weights.gender * instance.gender)
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
    # TODO: Fill in your code here
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

    # For each data point:
    while dataset.hasNext():
      instance = dataset.nextInstance()

      # Your code: perform delayed regularization

      # Your code: predict the label, record the loss
      if (mod(dataset.counter,output_steps)==0 and dataset.counter > 0) or (!dataset.hasNext()):
        if dataset.hasNext():
          (predictions,avg_loss[dataset.counter / output_steps]) = self.predict(weights,dataset)
          print "Average loss after %d iterations: %f" %(dataset.counter, avg_loss[dataset.counter / output_steps])
        else:
          (predictions,avg_loss[(dataset.counter/output_steps) + 1]) = self.predict(weights,dataset)
          print "Average loss after %d iterations: %f" %(dataset.counter, avg_loss[(dataset.counter/output_steps)+1])

        
      # Compute w0 + <w, x>, and gradient
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights, instance))
      coeff = instance.clicked - exp_inner_product / (1 + exp_inner_product)
      
      # Update weights along the negative gradient
      weights.w_age -= step * instance.age * coeff
      weights.w_gender -= step * instance.gender * coeff
      weights.w_depth -= step * instance.depth * coeff
      weights.w_position -= step * instance.position * coeff
      weights.w_w0 -= step * coeff

      
    dataset.reset()
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    predictions = [None] * dataset.size
    count = dataset.counter             # Record position in dataset so it can be returned the same way
    loss = 0.0                          # Compute loss as we go if we are looking at the training set

    # Check if we're working with training data
    if dataset.has_label:
      # Get predictions for the later parts of dataset
      while dataset.hasNext():
        instance = dataset.nextInstance()
        exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
        predictions[dataset.counter] = exp_inner_product / (1 + exp_inner_product)
        loss += pow(predictions[dataset.counter] - instance.clicked,2)

      # Get predictions for the earlier part of the dataset
      dataset.reset()
      while dataset.counter < count:
        instance = dataset.nextInstance()
        exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
        predictions[dataset.counter] = exp_inner_product / (1 + exp_inner_product)
        loss += pow(predictions[dataset.counter] - instance.clicked,2)

    else:
      # Get predictions for the later parts of dataset
      while dataset.hasNext():
        instance = dataset.nextInstance()
        exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
        predictions[dataset.counter] = exp_inner_product / (1 + exp_inner_product)

      # Get predictions for the earlier part of the dataset
      dataset.reset()
      while dataset.counter < count:
        instance = dataset.nextInstance()
        exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
        predictions[dataset.counter] = exp_inner_product / (1 + exp_inner_product)


    return (predictions,loss / count)
  
  
if __name__ == '__main__':
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552


  # Hyperparameters:
  lambduh = 0           # Regularization parameter
  step = 1.e-3               # SGD stepsize

  training = DataSet("../../data/train.txt", True, TRAININGSIZE)
  avg_loss = [None] * (TRAININGSIZE / 100 + 1)
  print "Training Logistic Regression..."

