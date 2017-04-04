from analysis.DataInstance import DataInstance
from analysis.HashedDataInstance import HashedDataInstance
from util.HashUtil import HashUtil

# This class represents a dataset object.
class DataSet:
  TRAININGSIZE = 2335859
  TESTINGSIZE = 1016552
  
  # ==========================
  # Creates a dataset from the given path.
  # 
  # @param path {String} Path to the data file living on the disk.
  # @param is_training {Boolean} True if the input is training data.
  # @param size {Int} The size of the dataset, can be SMALLER than the
  #                   size of the input.
  # ==========================
  def __init__(self, path, is_training, size):
    self.path = path
    self.has_label = is_training
    self.size = size
    self.counter = 0
    self.file_handler = open(path, 'r')

  # ==========================
  # @return true if the dataset has more data.
  # ==========================
  def hasNext(self):
    return self.counter < self.size
  
  # ==========================
  # @return the next data instance.
  # ==========================
  def nextInstance(self):
    line = self.file_handler.readline()
    self.counter += 1
    return DataInstance(line, self.has_label)
  
  # ==========================
  # @param featuredim {Int}
  # @param personal {Boolean}
  # @return the next data instance with hashed features.
  # ==========================
  def nextHashedInstance(self, featuredim, personal):
    line = self.file_handler.readline()
    self.counter += 1
    return HashedDataInstance(line, self.has_label, featuredim, personal)
  
  # ==========================
  # Reset the dataset. Must be called when ever the same dataset need to be
  # reused.
  # ==========================
  def reset(self):
    self.counter = 0
    self.file_handler.close()
    self.file_handler = open(self.path, 'r')
