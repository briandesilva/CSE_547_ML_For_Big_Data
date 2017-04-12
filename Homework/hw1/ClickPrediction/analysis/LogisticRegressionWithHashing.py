import math
import time       # For timing code
import matplotlib.pyplot as plt


from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class Weights:
  def __init__(self, featuredim):
    self.featuredim = featuredim
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # hashed feature weights
    self.w_hashed_features = [0.0 for _ in range(featuredim)]
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
    string += "Hashed Feature: "
    string += " ".join([str(val) for val in self.w_hashed_features])
    string += "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_hashed_features:
      l2 += w * w
    return math.sqrt(l2)


class LogisticRegressionWithHashing:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    # First few components of inner product
    output =  1.0 * (weights.w0 + weights.w_age * instance.age + weights.w_depth * instance.depth
      + weights.w_position * instance.position + weights.w_gender * instance.gender)
    # Components of inner product involving tokens
    for i in instance.hashed_text_feature:
      output += weights.w_hashed_features[i]

    return output
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param featureids {[Int]} list of feature ids
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, featureids, weights, now, step, lambduh):
    # h = HashUtil()
    for fid in featureids:
      if weights.w_hashed_features[fid] != 0.0:
        weights.w_hashed_features[fid] *= pow(1. - step * lambduh, now - weights.access_time[fid])
      weights.access_time[fid] = now
    return
  
  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, dim, lambduh, step, avg_loss, personalized):
    weights = Weights(dim)

    # Number of steps to perform between each update
    output_steps = 100

    # Variable to record the running loss
    loss = 0.0

    # For each data point:
    while dataset.hasNext():
      instance = dataset.nextHashedInstance(dim, personalized)

      # Perform delayed regularization
      self.perform_delayed_regularization(instance.hashed_text_feature,weights,dataset.counter,step,lambduh)

      # Predict the label, record the loss
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
      prediction = exp_inner_product / (1 + exp_inner_product)

      loss += pow(prediction - instance.clicked, 2)


      if ((dataset.counter % output_steps)==0) or (not dataset.hasNext()):
        if dataset.hasNext():
          avg_loss[(dataset.counter / output_steps) - 1] = loss / dataset.counter
        else:
          avg_loss[dataset.counter / output_steps] = loss / dataset.counter

        
      # Compute w0 + <w, x>, and gradient
      coeff = instance.clicked - prediction
      
      # Update weights along the negative gradient
      weights.w_age += step * (instance.age * coeff - lambduh * weights.w_age)
      weights.w_gender += step * (instance.gender * coeff - lambduh * weights.w_gender)
      weights.w_depth += step * (instance.depth * coeff - lambduh * weights.w_depth)
      weights.w_position += step * (instance.position * coeff - lambduh * weights.w_position)
      weights.w0 += step * coeff

      # Update weights corresponding to tokens
      for i in instance.hashed_text_feature:
        weights.w_hashed_features[i] += step * coeff

    # Perform delayed regularization on all the weights
    self.perform_delayed_regularization(range(dim),weights,dataset.counter,step,lambduh)
      
    dataset.reset()
    return weights

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # @param personalized {Boolean}
  # ==========================
  def predict(self, weights, dataset, personalized):
    predictions = [None] * dataset.size
    count = dataset.counter             # Record position in dataset so it can be returned the same way

    # Get predictions for the later parts of dataset
    while dataset.hasNext():
      instance = dataset.nextHashedInstance(weights.featuredim,personalized)
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
      predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)

    # Get predictions for the earlier part of the dataset
    dataset.reset()
    while dataset.counter < count:
      instance = dataset.nextHashedInstance(weights.featuredim,personalized)
      exp_inner_product = math.exp(self.compute_weight_feature_product(weights,instance))
      predictions[dataset.counter-1] = exp_inner_product / (1 + exp_inner_product)


    return predictions
  
  
if __name__ == '__main__':

    TRAININGSIZE = 2335859
    TESTINGSIZE = 1016552

    training = DataSet("../../data/train.txt", True, TRAININGSIZE)
    testing = DataSet("../../data/test.txt", False, TESTINGSIZE)

    # Hyperparameters:

    lambduh = 0.001
    step = 0.01
    m = [101, 12277, 1573549]
    personalized = False

    avg_loss = [None] * (TRAININGSIZE / 100 + 1)
    weight_vec = [None] * len(m)
    test_rmse_vec = [None] * len(m)

    # Set up some objects we will need
    l = LogisticRegressionWithHashing()
    evaluator = EvalUtil()


    # Train the logistic regression model for various stepsizes
    for k in range(len(m)):
      t0 = time.time()
      print "Training Logistic Regression with Hashed Features..."
      weights = l.train(training,m[k],lambduh,step,avg_loss,personalized)
      print "Done.\nAverage loss on the training set (eta,lambda,m) = (%f,%f,%d): %f" %(step,lambduh,m[k],avg_loss[-1])
      weight_vec[k] = weights.l2_norm()
      print "l2 norm of weights (eta,lambda,m) = (%f,%f,%d): %f" %(step,lambduh,m[k],weights.l2_norm())
      print "Time used to train model for m = %d: %f" %(m[k], time.time() - t0)

      # Get the RMSE on the test data using logistic regression
      test_predictions = l.predict(weights,testing,personalized)

      test_rmse = evaluator.eval("../../data/test_label.txt",test_predictions)
      test_rmse_vec[k] = test_rmse
      print "RMSE on the test set using logistic regression (eta,lambda,m) = (%f,%f,%d): %f" %(step,lambduh,m[k],test_rmse)

    # Plot the RMSE as a function of the dimension of the hashed feature space
plt.semilogx(m,test_rmse_vec,'b-o')
plt.xlabel('Hashed feature space dimension')
plt.ylabel('RMSE')
plt.title('RMSE for predicted CTR on test set as hashed feature space dimension is increased')
plt.show()