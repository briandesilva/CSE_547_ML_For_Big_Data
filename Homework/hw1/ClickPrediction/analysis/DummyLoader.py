from analysis.DataSet import DataSet

class DummyLoader:
  # ==========================
  # Scan the data and print out to the stdout
  # @param dataset {DataSet}
  # ==========================
  def scan_and_print(self, dataset):
    count = 0
    print "Loading data from " + dataset.path + "..."
    while dataset.hasNext():
      instance = dataset.nextInstance()
      # Here we printout the instance. But your code for processing each
      # instance will come here. For example: tracking the max clicks,
      # update token frequency table, or compute gradient for logistic
      # regression...
      print str(instance)
      count += 1
      if count % 1000000 is 0:
        print "Loaded " + count + " lines"
    if count < dataset.size:
      print "Warning: the real size of the data is less than the input size: %d < %d" % (dataset.size, count)
    print "Done. Total processed instances: %d" % count
    
    # Important: remember to reset the dataset everytime
    # you are done with processing.
    dataset.reset()

if __name__ == '__main__':
  loader = DummyLoader()
  size = 10
  
  # prints a dataset from the training data with size = 10;
  training = DataSet("../../data/train.txt", True, size)
  loader.scan_and_print(training)
  
  # prints a dataset from the test data with size = 10;
  testing = DataSet("../../data/test.txt", False, size)
  loader.scan_and_print(testing)