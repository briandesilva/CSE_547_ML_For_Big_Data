# This class represents an instance of the data.
class DataInstance:
  def __init__(self, line, has_label):
    fields = line.split("|")
    offset = 0
    
    if has_label:
      # whether clicked: 0 or 1
      self.clicked = int(fields[0])
      offset = 1
    else:
      self.clicked = -1

    # depth of the session
    self.depth = int(fields[offset])
    # position of the ad
    self.position = int(fields[offset + 1])
    self.userid = int(fields[offset + 2])
    # user gender indicator (-1 for male, 1 for female)
    self.gender = int(fields[offset+3])
    if self.gender != 0:
      self.gender = int((self.gender - 1.5) * 2)
    # user age indicator:
    #   '1' for (0, 12],
    #   '2' for (12, 18],
    #   '3' for (18, 24],
    #   '4' for (24, 30],
    #   '5' for (30, 40], and
    #   '6' for greater than 40.
    self.age = int(fields[offset + 4])
    # list of token ids
    self.tokens = [int(xx) for xx in fields[offset + 5].split(",")]
    
  def __str__(self):
    string = ""
    if self.clicked >= 0:
      string += str(self.clicked) + "|"
    string += str(self.depth) + "|"
    string += str(self.position) + "|"
    string += str(self.userid) + "|"
    string += str(self.gender) + "|"
    string += str(self.age) + "|"
    string += ",".join([str(tok) for tok in self.tokens])
    return string