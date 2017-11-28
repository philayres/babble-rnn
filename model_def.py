import keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Concatenate, Input
from keras.layers import GRU, LSTM, Conv2D, Conv1D, Reshape, Flatten, Permute, AveragePooling2D, MaxPooling2D, RepeatVector
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

    in_scale = 5
    in_count = framelen * in_scale

    for i in range(0, in_count):

        d0 = TimeDistributed(
            Dense(
                    3
                    , activation="relu"
                    , trainable=True
            )
        )(main_input)

        d005 = TimeDistributed(
            Dense(
                    5
                    , activation="relu"
                    , trainable=True
            )
        )(d0)

        d01 = TimeDistributed(
            Dense(
                    3
                    , activation="relu"
                    , trainable=True
            )
        )(d005)

        l0.append(d01)
        # l0.append(
        #     LSTM(
        #             3
        #             , return_sequences=True
        #             , trainable=True
        #     )(d0)
        # )


    for i in range(0, in_count):
        j = i - 1
        if j < 0:
            j = in_count - 1
        cl = keras.layers.concatenate([l0[j], l0[i]])
        # l01 = LSTM(
        #             6
        #             , return_sequences=True
        #             , trainable=True
        # )(cl)

        l001 = TimeDistributed(
            Dense(
                framelen
                , activation="relu"
                , trainable=True
                )
            )(cl)

        l01 = TimeDistributed(
            Dense(
                framelen * 19
                , activation="relu"
                , trainable=True
                )
            )(l001)

        lout.append(
            TimeDistributed(
                Dense(
                    framelen
                    , activation="relu"
                    , trainable=True
                    )
                )(l01)
        )

    lout.append(main_input)

    conc = keras.layers.concatenate(lout)

    lmid = LSTM(
        framelen
        , return_sequences=False
        , trainable=False
    )(conc)
    mid_output = Dense(framelen, name="mid_output")(lmid)

    # bring this back down to size...
    cd0 = TimeDistributed(
        Dense(
            in_count
            , activation="relu"
            , trainable=True
        )
    )(conc)

    cr = TimeDistributed(keras.layers.Reshape((in_count, 1)))(cd0)

    conv0 = Conv2D(in_count, 5, padding='same', data_format='channels_last', use_bias=True
    )(cr)

    mp0 = MaxPooling2D(in_scale, padding='valid', data_format='channels_last')(conv0)

    # Need to repeat here
    rs1 = TimeDistributed(keras.layers.Reshape((framelen)))(mp0)

    rp0 = RepeatVector(in_scale)(rs1)

    rp = keras.layers.Reshape((framelen))(rp0)

    recomb = concatenate([rp, main_input])

    l20 = LSTM(
        framelen * 10
        , return_sequences=True
        , trainable=True
        , name='LSTM_post_mid_1'
    )(recomb)

    # cd = TimeDistributed(Dense(
    # framelen * 12
    # , trainable=True
    # ))(l20)

    # l21 = LSTM(
    #     framelen * 10
    #     , return_sequences=True
    #     , trainable=True
    # )(l20)


    l2 = LSTM(
        framelen * 10
        , return_sequences=False
        , trainable=True
    )(l20)


    main_output = Dense(
      framelen
      ,activation="relu"
      , trainable=True
      , name="main_output"
    )(l2)



    model = Model(
        inputs=[main_input],
        outputs=[main_output, mid_output]
    )

    self.model = model
    return model


  def compile_model(self):
    self.utils.log("Compiling model")

    loss = CustomObjects.codec2_param_error
    # other loss options: CustomObjects.codec2_param_mean_square_error; 'mean_absolute_error'; 'cosine_proximity'

    self.model.compile(
        loss={'main_output': loss, 'mid_output': loss},
        loss_weights={'main_output': 1., 'mid_output': 0.2},
        optimizer=self.get_optimizer_from_config())
    self.utils.log_model_summary()

  def fit(self, input_seq, output_seq, batch_size=None, epochs=1, shuffle=False, callbacks=None):
      inputs = input_seq
      outputs = {'main_output': output_seq, 'mid_output': output_seq}
      self.model.fit(inputs, outputs, batch_size=batch_size, epochs=epochs, shuffle=shuffle,
       callbacks=callbacks
      )





  # we can wrap the model.add method, since in the future we may wish to
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
#
#    elif iteration == 481:
#      self.model_updates_lstm3_trainable()
#


    if not self.started:
      self.model_updates_onstart()


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




  def load_weights(self, fn, by_name=False):
    self.utils.log("Loading weights")
    self.model.load_weights(fn, by_name=by_name)

  def get_optimizer_from_config(self):
      optimizer_name = self.config.optimizer["name"]
      args = []
      self.optimizer = getattr(optimizers, optimizer_name)(*args, **self.config.optimizer["params"])
      return self.optimizer
