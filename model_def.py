from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed
from keras.layers import GRU, LSTM
import keras.optimizers as optimizers

from custom_objects import CustomObjects

class ModelDef(object):
  custom_objects = CustomObjects()

  layers=[]

  model = None
  utils = None
  started = False
  
# for 1300 rate codec  
#  frame_property_scaleup = [
#   1,1,1,1,
#   2**7,
#   2**5,
#   16,16,16,16,16,16,16,8,8,4
#  ]

# for 3200 rate codec
  frame_property_scaleup = CustomObjects.frame_prop_loss_scale
#  [
#   1,
#   2**7,
#   2**5,
#   32,32,32,32,32,32,32,32,32,32
#  ]

  stateful = False
  

  def __init__(self, utils, config):
    self.utils = utils
    self.config = config
    
    self.layers=[]
    
    utils.log("frame_property_scaleup: ", self.frame_property_scaleup)

  
  def define_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.utils.log("Defining model")
    model =  Sequential()
    self.model = model

    if self.stateful:
        self.add_layer(
          LSTM(
            320
            , batch_input_shape=(num_frame_seqs , frame_seq_len, framelen) 
            , return_sequences=True
            , trainable=True
            , stateful=self.stateful
        #    ,dropout = 0.1
          )
        )
    else:
        self.add_layer(
          LSTM(
            320
            , input_shape=(frame_seq_len, framelen) 
            , return_sequences=True
            , trainable=True
            , stateful=self.stateful
        #    ,dropout = 0.1
          )
        )

    self.add_layer(
      LSTM(
        320
        , return_sequences=True
        , trainable=False
        , stateful=self.stateful
    #    ,dropout = 0.1
        
      )
    )
    
    
    self.add_layer(
      LSTM(
        320
        , return_sequences=True
        , trainable=False
        , stateful=self.stateful
    #    ,dropout = 0.1
      )
    )
    
    
    self.add_layer(
      TimeDistributed(
        Dense(
          framelen
          ,activation="elu"
        )
      )
    )
    #model.add(Dropout(0.1))
    
    return model

  # we wrap the model.add method, since in the future we may wish to 
  # provide additional processing at this level
  def add_layer(self, layer):
    self.model.add(layer)
    return layer
    

  # start training GRU 1, then 1&2, then 3 
  def before_iteration(self, iteration):
    if not self.started:
      self.model_updates_onstart()
      self.started = True
    
#    elif iteration == 121:
#      self.model_updates_lstm12_trainable()
#
#    elif iteration == 481:
#      self.model_updates_lstm3_trainable()
#      
  def model_updates_onstart(self):
    self.model_updates_lstm_123_trainable()
    #self.model_updates_lstm1_trainable()  
  
  def model_updates_lstm_123_trainable(self):
    self.utils.log("Make lstm 1,2,3 trainable")
    self.model.layers[0].trainable=True
    self.model.layers[1].trainable=True
    self.model.layers[2].trainable=True
    self.compile_model()
    self.utils.save_json_model(4)

  def model_updates_lstm_23_trainable(self):
    self.utils.log("Make lstm 2,3 trainable")
    self.model.layers[0].trainable=False
    self.model.layers[1].trainable=True
    self.model.layers[2].trainable=True
    self.compile_model()
    self.utils.save_json_model(4)


  def model_updates_lstm2_trainable(self):
    self.utils.log("Make lstm 2 trainable")
    self.model.layers[0].trainable=False
    self.model.layers[1].trainable=True
    self.model.layers[2].trainable=False
    self.compile_model()
    self.utils.save_json_model(1)
  
  def model_updates_lstm3_trainable(self):
    self.utils.log("Make lstm 3 trainable")
    self.model.layers[0].trainable=False
    self.model.layers[1].trainable=False
    self.model.layers[2].trainable=True
    self.compile_model()
    self.utils.save_json_model(2)

  def model_updates_lstm1_trainable(self):
    self.utils.log("Make lstm 1 trainable")
    self.model.layers[0].trainable=True
    self.model.layers[1].trainable=False
    self.model.layers[2].trainable=False
    self.compile_model()
    self.utils.save_json_model(3)
  
  def model_updates_lstm12_trainable(self):
    self.utils.log("Make lstm 1 & 2 trainable")
    self.model.layers[0].trainable=True
    self.model.layers[1].trainable=True
    self.model.layers[2].trainable=False
    self.compile_model()
    self.utils.save_json_model(3)

    
  def load_weights(self, fn, by_name=False):
    self.utils.log("Loading weights")
    self.model.load_weights(fn, by_name=by_name)
    
  def compile_model(self):
    self.utils.log("Compiling model")
    
    optimizer_name = self.config.optimizer["name"]
    
    optimizer = getattr(optimizers, optimizer_name)(*self.config.optimizer["params"])
      #optimizer = Nadam() #SGD() #Adam() #RMSprop(lr=0.01)
    
    
    loss = CustomObjects.codec2_param_error
    #loss = 'mean_absolute_error'
    #loss = 'cosine_proximity'
    self.model.compile(loss=loss, optimizer=optimizer)
    
