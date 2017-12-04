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

  encoder_trainable = True
  decoder_trainable = True
  generator_trainable = False

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


    print("short_input_len", short_input_len)

    main_input = Input(shape=(frame_seq_len, framelen), dtype='float32', name="main_input")
    # if overlap_sequence != 0:
    short_input = Input(shape=(short_input_len, framelen), dtype='float32', name="short_input")

    lout = []
    l0 = []

    # cin = keras.layers.concatenate([short_input, short_input])

    # rpd0 = TimeDistributed(Dense(conv_count, trainable=encoder_trainable))(cin)
    # rpd = TimeDistributed(Dense(conv_count, trainable=encoder_trainable))(rpd0)

    rpd = short_input
    # Attempt to the decoder back to the original input


    lmid = LSTM(
        framelen * 10
        , return_sequences=False
        , trainable=decoder_trainable
    )(short_input)
    # mid_d0 = Dense(framelen, trainable=decoder_trainable)(short_input)
    mid_output = Dense(framelen, name="mid_output", trainable=decoder_trainable)(lmid)




    recomb = keras.layers.concatenate([rpd, short_input])

    l20 = LSTM(
        framelen * 10
        , return_sequences=True
        , name='LSTM_post_mid_1'
        , trainable=generator_trainable
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
        , trainable=generator_trainable
    )(l20)


    main_output = Dense(
      framelen
      , activation="relu"
      , trainable=generator_trainable
      , name="main_output"
    )(l2)



    model = Model(
        inputs=[main_input, short_input],
        outputs=[main_output, mid_output]
    )


    self.model = model
    return model



  def define_model_bak(self, frame_seq_len, framelen, num_frame_seqs):
    self.utils.log("Defining model")
    config = self.config
    overlap_sequence = config.overlap_sequence
    short_input_len = frame_seq_len - overlap_sequence*2
    in_scale = 2
    in_count = framelen * in_scale
    conv_count = 65

    print("short_input_len", short_input_len)

    main_input = Input(shape=(frame_seq_len, framelen), dtype='float32', name="main_input")
    # if overlap_sequence != 0:
    short_input = Input(shape=(short_input_len, framelen), dtype='float32', name="short_input")

    lout = []
    l0 = []

    cin = keras.layers.concatenate([main_input, main_input])

    encoder_trainable = True

    cr = TimeDistributed(keras.layers.Reshape((in_count, 1), trainable=encoder_trainable))(cin)

    conv0_def = Conv2D(conv_count, (1,14), padding='valid', data_format='channels_last', trainable=encoder_trainable)
    conv0 = conv0_def(cr)

    conf = conv0_def
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    # mp = MaxPooling2D(2, padding='valid', data_format='channels_last')
    # mp0 = mp(conv0)


    conv1_def = Conv2D(conv_count, (5,13), padding='valid', data_format='channels_last', trainable=encoder_trainable)
    conv1 = conv1_def(conv0)

    conf = conv1_def
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)





    # td0_conf =  TimeDistributed(keras.layers.Reshape((short_input_len, conv_count)))
    # td0 = td0_conf(conv1)
    # conf = td0_conf
    # print(conf.get_config())
    # print(conf.input_shape)
    # print(conf.output_shape)


    # # rpl = TimeDistributed(RepeatVector(15))
    # # # Need to repeat here
    # # rp0 = rpl(rs1)
    # # conf = rpl
    # # print(conf.get_config())
    # # print(conf.input_shape)
    # # print(conf.output_shape)
    #
    rs0 = keras.layers.Reshape((short_input_len, conv_count), trainable=encoder_trainable)(conv1)

    rpd0 = TimeDistributed(Dense(conv_count, trainable=encoder_trainable))(rs0)
    rpd = TimeDistributed(Dense(conv_count, trainable=encoder_trainable))(rpd0)


    # Attempt to the decoder back to the original input

    decoder_trainable = True

    lmid = LSTM(
        framelen * 10
        , return_sequences=False
        , trainable=decoder_trainable
    )(rpd)
    mid_d0 = Dense(framelen, trainable=decoder_trainable)(lmid)
    mid_output = Dense(framelen, name="mid_output", trainable=decoder_trainable)(mid_d0)




    recomb = keras.layers.concatenate([rpd, short_input])

    generator_trainable = False

    l20 = LSTM(
        framelen * 10
        , return_sequences=True
        , name='LSTM_post_mid_1'
        , trainable=generator_trainable
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
        , trainable=generator_trainable
    )(l20)


    main_output = Dense(
      framelen
      , activation="relu"
      , trainable=generator_trainable
      , name="main_output"
    )(l2)



    model = Model(
        inputs=[main_input, short_input],
        outputs=[main_output, mid_output]
    )

    self.generator_trainable = generator_trainable
    self.decoder_trainable = decoder_trainable

    self.model = model
    return model


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
        loss={'main_output': loss, 'mid_output': loss},
        loss_weights={'main_output': main_loss_prop, 'mid_output': mid_loss_prop},
        optimizer=self.get_optimizer_from_config())
    self.utils.log_model_summary()

  def fit(self, input_seq, output_seq, batch_size=None, epochs=1, shuffle=False, callbacks=None):
      inputs = input_seq

    #   if self.config.overlap_sequence == 0:
    #       outputs = {'main_output': output_seq, 'mid_output': output_seq}
    #   else:
      # Attempt to learn the mid output as a decoder of the original input
      outputs = {'main_output': output_seq[0], 'mid_output': output_seq[1]}

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
