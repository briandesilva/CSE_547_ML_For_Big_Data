from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    # Store unique tokens
    tokens = set()
    while dataset.hasNext():
      instance = dataset.nextInstance()
      tokens.update(instance.tokens)

    dataset.reset()
    return tokens
  
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    users = set()
    while dataset.hasNext():
      instance = dataset.nextInstance()
      users.update(instance.userid)

    dataset.reset()
    return users

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    # Create dictionary in which to store unique user ids for each age group
    age_groups = {0:set(), 1:set(), 2:set(), 3:set(), 4:set(), 5:set(), 6:set()}
    while dataset.hasNext():
      instance = dataset.nextInstance()
      age_groups[instance.age].add(instance.userid)
    dataset.reset()
    return age_groups

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    total_clicks = 0.

    if dataset.has_label:                     # Make sure the dataset has labels
      while dataset.hasNext():                # Loop over instances
        instance = dataset.nextInstance()
        total_clicks += instance.clicked
      dataset.reset()
      return total_clicks / dataset.size
    else:
      return -1

if __name__ == '__main__':

  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552

  # Form a BasicAnalysis object
  analyze = BasicAnalysis()

  # # Load the training set
  training = DataSet("../../data/train.txt", True, TRAININGSIZE)

  # Load the testing set
  testing = DataSet("../../data/test.txt", False, TESTINGSIZE)

  # # Find the average CTR and print it out
  # print "Average CTR: %f"%analyze.average_ctr(training)

  # # Find the number of unique tokens in the training data and print it out
  # training_tokens = analyze.uniq_tokens(training)
  # print "Number of unique tokens in the training data: %d" %len(training_tokens)

  # # Find the number of unique tokens in the testing data and print it out
  # testing_tokens = analyze.uniq_tokens(testing)
  # print "Number of unique tokens in the testing data: %d" %len(testing_tokens)

  # # Find the number of unique tokens in both datasets and print it out
  # print "Number of tokens present in both datasets: %d" %len(training_tokens.intersection(testing_tokens))

  # Find the number of unique users in each age group in the training data and print it out
  training_user_dict = analyze.uniq_users_per_age_group(training)
  for i in range(7):
    print "Number of unique users in age group %d in training set: %d" %(i,len(training_user_dict[i]))

  # Find the number of unique users in each age group in the testing data and print it out
  testing_user_dict = analyze.uniq_users_per_age_group(testing)
  for i in range(7):
    print "Number of unique users in age group %d in testing set: %d" %(i,len(testing_user_dict[i]))


