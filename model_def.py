import keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Concatenate, Input
from keras.layers import GRU, LSTM, Conv2D, Conv1D, Reshape, Flatten, Permute, AveragePooling2D, MaxPooling2D, RepeatVector, Conv2DTranspose
import keras.optimizers as optimizers

from custom_objects import CustomObjects

class ModelDef(object):

  layers=[]

  model = None
  utils = None
  started = False


  stateful = False

  encoder_trainable = True
  decoder_trainable = True
  generator_trainable = True
  decoder_model_memo = None

  def __init__(self, utils, config):
    self.utils = utils
    self.config = config

    self.layers=[]




  def define_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.utils.log("Defining model")
    config = self.config
    overlap_sequence = config.overlap_sequence
    short_input_len = frame_seq_len - overlap_sequence*2
    in_scale = 2
    in_count = framelen * in_scale
    conv_count = 65

    encoder_trainable = self.encoder_trainable
    decoder_trainable = self.decoder_trainable
    generator_trainable = self.generator_trainable
    self.utils.log("encoder_trainable:", self.encoder_trainable)
    self.utils.log("decoder_trainable:", self.decoder_trainable)
    self.utils.log("generator_trainable:", self.generator_trainable)

    print("short_input_len", short_input_len)

    main_input = Input(shape=(frame_seq_len, framelen), dtype='float32', name="main_input")
    # short_input = Input(shape=(short_input_len, framelen), dtype='float32', name="short_input")

    cin = keras.layers.concatenate([main_input, main_input])

    cr = TimeDistributed(keras.layers.Reshape((in_count, 1), trainable=encoder_trainable))(cin)

    conv0_def = Conv2D(conv_count, (1,14), padding='valid', data_format='channels_last', activation='relu', trainable=encoder_trainable)
    conv0 = conv0_def(cr)

    conf = conv0_def
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conv1_def = Conv2D(conv_count, (3,13), strides=(2,1), padding='valid', data_format='channels_last', activation='relu', trainable=encoder_trainable)
    conv1 = conv1_def(conv0)

    conf = conv1_def
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)



    rs0 = keras.layers.Reshape((-1, conv_count), trainable=encoder_trainable)(conv1)

    conf = TimeDistributed(
        Dense(
            conv_count
            , activation="relu"
            , trainable=encoder_trainable
        )
    )
    encoder_output = conf(rs0)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)



    # Run the decoder portion of autoencoder
    mid_output = self.decoder_model(framelen, (-1, conv_count))(encoder_output)



    # Generator


    l20 = LSTM(
        framelen * 10
        , return_sequences=True
        , name='generator_LSTM_0'
        , trainable=generator_trainable
    )(encoder_output)


    l2 = LSTM(
        framelen * 10
        , return_sequences=True
        , name='generator_LSTM_1'
        , trainable=generator_trainable
    )(l20)


    conf = TimeDistributed(
        Dense(
          conv_count
          , activation="relu"
          , trainable=generator_trainable
        )
        , name='generator_TD_Dense_0'
    )
    generator_output = conf(l2)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    main_output = self.decoder_model(framelen, (-1, conv_count))(generator_output)


    model = Model(
        #inputs=[main_input, short_input],
        inputs=[main_input],
        outputs=[main_output, mid_output]
    )

    self.model = model
    return model

  # Shared Decoder model, taking Conv2D encoded input, or an attempt at predicted
  # sequences of the encoded data.
  #
  # Returns a single instance of a decoder model. May be called any number of times.
  #
  # Arguments:
  #  framelen: standard frame length for a timestep
  #  shape: two element tensor, typically (-1, conv_count), where conv_count is the
  #         width of the input timestep. -1 saves us having to enter the
  #         frame sequence (sub-batch) length
  #
  # Learning can be disabled by setting self.decoder_trainable = False
  #
  # Example:
  #    main_output = self.decoder_model(framelen, (-1, conv_count))(generator_output)
  def decoder_model(self, framelen, shape=(-1)):
    conv_count = shape[1]
    if self.decoder_model_memo:
      return self.decoder_model_memo

    decoder_input = Input(shape=shape, dtype='float32', name="decoder_input")

    cr = TimeDistributed(
            keras.layers.Reshape(
              (1, conv_count)
              , trainable=self.decoder_trainable
            )
          )(decoder_input)

    conf = Conv2DTranspose(
              conv_count,
              kernel_size=(3,13),
              padding='valid',
              strides=(4,14),
              activation='relu',
              data_format="channels_last",
              trainable=self.decoder_trainable
    )
    decoder_deconv_0 = conf(cr)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    # conf = Conv2DTranspose(
    #   conv_count,
    #   kernel_size=(1,14),
    #   padding='valid',
    #   strides=(1,2),
    #   activation='relu',
    #   data_format="channels_last",
    #   trainable=self.decoder_trainable
    # )
    #
    # decoder_deconv_1 = conf(decoder_deconv_0)
    #
    # print(conf.get_config())
    # print(conf.input_shape)
    # print(conf.output_shape)

    conf = Conv2D(1,
                   kernel_size=2,
                   padding='valid',
                   activation='sigmoid')
    decoder_mean_squashed = conf(decoder_deconv_0)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf  = keras.layers.Reshape((-1, framelen), trainable=self.decoder_trainable)
    rs0 = conf(decoder_mean_squashed)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)


    lmid = LSTM(
        framelen * 3
        , return_sequences=True
        , trainable=self.decoder_trainable
    )(rs0)


    decoder_output = TimeDistributed(
        Dense(
            framelen
            , activation="relu"
            , trainable=self.decoder_trainable
        )
    )(lmid)

    self.decoder_model_memo = Model(decoder_input, decoder_output)
    return self.decoder_model_memo



  def compile_model(self):
    self.utils.log("Compiling model")

    loss = CustomObjects.codec2_param_error
    # other loss options: CustomObjects.codec2_param_mean_square_error; 'mean_absolute_error'; 'cosine_proximity'

    main_loss_prop = 0.5
    mid_loss_prop = 0.5

    if not self.generator_trainable and self.decoder_trainable:
      main_loss_prop = 0
      mid_loss_prop = 1
    elif not self.decoder_trainable and self.generator_trainable:
      mid_loss_prop = 0
      main_loss_prop = 1



    self.model.compile(
        loss=[loss, loss], #{'main_output': loss, 'mid_output': loss},
        loss_weights=[main_loss_prop, mid_loss_prop],#{'main_output': main_loss_prop, 'mid_output': mid_loss_prop},
        optimizer=self.get_optimizer_from_config())
    self.utils.log_model_summary()

  def fit(self, input_seq, output_seq, batch_size=None, epochs=1, shuffle=False, callbacks=None):
      inputs = input_seq[0]

      #outputs = {'main_output': output_seq[0], 'mid_output': output_seq[1]}
      outputs = output_seq
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
