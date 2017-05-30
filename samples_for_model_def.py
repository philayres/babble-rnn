######
# samples for model_def to provide alternative network updates on interation 
#####


  # a version that flips the 'trainable' between LSTM 1&2 and 2&3 every iteration 
  def before_iteration(self, iteration):
    if not self.started:
      self.model_updates_onstart()
      self.started = True    
    elif iteration % 2 == 0:
      self.model_updates_lstm12_trainable()
    else:
      self.model_updates_lstm_23_trainable()

  def model_updates_onstart(self):
    self.model_updates_lstm12_trainable()
      
  # a version to rotate through trainable LSTMs every 10 iterations
  # start with 1 being trainable, then 2, then 3    
  def before_iteration(self, iteration):
    if not self.started:
      self.model_updates_onstart()
      self.started = True
    
    elif (iteration-1) % 30 == 0:
      self.model_updates_lstm1_trainable()

    
    elif (iteration-1) % 20 == 0:
      self.model_updates_lstm3_trainable()

    elif (iteration-1) % 10 == 0:
      self.model_updates_lstm2_trainable()

  def model_updates_onstart(self):
    self.model_updates_lstm1_trainable()
    
