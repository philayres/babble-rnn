import keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Concatenate, Input
from keras.layers import GRU, LSTM, Conv2D, Conv1D, Reshape, Flatten, Permute, AveragePooling2D, MaxPooling2D
import keras.optimizers as optimizers

from custom_objects import CustomObjects

class ModelDef(object):

  layers=[]

  model = None
  utils = None
  started = False


  stateful = False


  def __init__(self, utils, config):
    self.utils = utils
    self.config = config

    self.layers=[]

  def define_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.utils.log("Defining model")

    main_input = Input(shape=(frame_seq_len, framelen), dtype='float32', name="main_input")

    lout = []
    l0 = []

    for i in range(0,13):

        d0 = TimeDistributed(
            Dense(
                    3
                    , activation="relu"
                    , trainable=True
            )
        )(main_input)

        l0.append(
            LSTM(
                    3
                    , return_sequences=True
                    , trainable=True
            )(d0)
        )


    for i in range(0,13):
        j = i - 1
        if j < 0:
            j = 12
        cl = keras.layers.concatenate([l0[j], l0[i]])
        l01 = LSTM(
                    6
                    , return_sequences=True
                    , trainable=True
        )(cl)

        lout.append(
            TimeDistributed(
                Dense(
                    6
                    , activation="relu"
                    , trainable=True
                    )
                )(l01)
        )

# I'd like to pull out individual, intermediate outputs here, to play with additional loss calculations on each,
# targeting different parts of the overall frame
# Also need to feed the original input in back at this level...

    c = keras.layers.concatenate(lout)

    lmid = LSTM(
        framelen
        , return_sequences=False
        , trainable=False
    )(c)
    mid_output = Dense(framelen, name="mid_output")(lmid)


    cd = Dense(
        framelen * 12
        , trainable=True
    )(c)

    l20 = LSTM(
        framelen * 10
        , return_sequences=True
        , trainable=True
    )(cd)

    l21 = LSTM(
        framelen * 10
        , return_sequences=True
        , trainable=False
    )(l20)


    l2 = LSTM(
        framelen * 10
        , return_sequences=False
        , trainable=False
    )(l21)


    main_output = Dense(
      framelen
      ,activation="relu"
      , trainable=True
      , name="main_output"
    )(l2)

    model = Model(inputs=[main_input], outputs=[main_output, mid_output])
    self.model = model
    return model







  # we wrap the model.add method, since in the future we may wish to
  # provide additional processing at this level
  def add_layer(self, layer):
    self.model.add(layer)
    return layer


  # start training GRU 1, then 1&2, then 3
  def before_iteration(self, iteration):
#    if iteration == 541:
#      self.utils.log("Adding frame rotation to reduce memory usage")
#      self.config.limit_frames = self.config.num_frames / 100
#      self.model_updates_lstm_1234_trainable()
#      self.config.log_attrs()


    if not self.started:
      self.model_updates_onstart()

#
#    elif iteration == 481:
#      self.model_updates_lstm3_trainable()
#

    self.started = True

  def model_updates_onstart(self):
#    self.utils.log("Make all lstms trainable")
#    self.model.layers[0].trainable=True
#    self.model.layers[1].trainable=True
#    self.model.layers[2].trainable=True
#    self.model.layers[3].trainable=True
#    self.model.layers[4].trainable=True

    self.compile_model()
    self.utils.save_json_model(0)


  def model_updates_lstm_1234_trainable(self):
    self.utils.log("Make lstm 1,2,3,4 trainable")
    self.model.layers[0].trainable=True
    self.model.layers[1].trainable=True
    self.model.layers[2].trainable=True
    self.model.layers[3].trainable=True
    self.compile_model()
    self.utils.save_json_model(4)

  def model_updates_lstm_123_untrainable(self):
    self.utils.log("Make lstm 1,2,3 untrainable ")
    self.model.layers[0].trainable=False
    self.model.layers[1].trainable=False
    self.model.layers[2].trainable=False
    self.compile_model()
    self.utils.save_json_model(5)

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
    args = []
    optimizer = getattr(optimizers, optimizer_name)(*args, **self.config.optimizer["params"])
      #optimizer = Nadam() #SGD() #Adam() #RMSprop(lr=0.01)


    #loss = CustomObjects.codec2_param_mean_square_error
    loss = CustomObjects.codec2_param_error
    #loss = 'mean_absolute_error'
    #loss = 'cosine_proximity'
    self.model.compile(
        loss={'main_output': loss, 'mid_output': loss},
        loss_weights={'main_output': 1., 'mid_output': 0.2},
        optimizer=optimizer)
    self.utils.log_model_summary()

  def fit(self, input_seq, output_seq, batch_size=None, epochs=1, shuffle=False, callbacks=None):
      inputs = input_seq
      outputs = {'main_output': output_seq, 'mid_output': output_seq}
      self.model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs, shuffle=shuffle,
       callbacks=callbacks
      )
