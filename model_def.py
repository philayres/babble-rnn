import keras as keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, TimeDistributed, Concatenate, Input, UpSampling2D, ZeroPadding2D, average, Lambda
from keras.layers import GRU, LSTM, Conv2D, Conv1D, Reshape, Flatten, Permute, AveragePooling2D, MaxPooling2D, RepeatVector, Conv2DTranspose, LocallyConnected2D
import keras.optimizers as optimizers
import numpy as np

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


  def __init__(self, utils, config):
    self.utils = utils
    self.config = config
    self.models = {}
    self.layers = []

    self.utils.log("encoder_trainable:", self.encoder_trainable)
    self.utils.log("decoder_trainable:", self.decoder_trainable)
    self.utils.log("generator_trainable:", self.generator_trainable)



  def define_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.utils.log("Defining model")
    config = self.config
    overlap_sequence = config.overlap_sequence
    short_input_len = frame_seq_len - overlap_sequence*2

    self.conv_count = 16
    enc_params = 15


    generator_trainable = self.generator_trainable

    print("short_input_len", short_input_len)

    main_input = Input(shape=(frame_seq_len, framelen), dtype='float32', name="main_input")
    # short_input = Input(shape=(short_input_len, framelen), dtype='float32', name="short_input")

    encoder_output = self.encoder_model(enc_params, shape=(frame_seq_len, framelen))(main_input)
    conf = self.encoder_model(framelen)
    print("encoder_model shapes for input / output 0")
    print(conf.get_input_shape_at(0))
    print(conf.get_output_shape_at(0))

    encoder_output_len = conf.get_output_shape_at(0)[1]

    # Run the decoder portion of autoencoder
    mid_output = self.decoder_model(framelen, (encoder_output_len, enc_params))(encoder_output)

    conf = self.decoder_model(framelen)
    print("decoder_model shapes for input / output 0")
    print(conf.get_input_shape_at(0))
    print(conf.get_output_shape_at(0))


    # Generator
    res = encoder_output
    res = TimeDistributed(Dropout(0.05))(res)

    conf = LSTM(
        128
        , return_sequences=True
        , name='generator_LSTM_0'
        , trainable=generator_trainable
        , kernel_initializer='he_normal'
        , recurrent_initializer='he_normal'
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    res = TimeDistributed(Dropout(0.05))(res)
    res = LSTM(
        128
        , return_sequences=True
        , name='generator_LSTM_1'
        , trainable=generator_trainable
        , kernel_initializer='he_normal'
        , recurrent_initializer='he_normal'
    )(res)

    res = TimeDistributed(Dropout(0.05))(res)
    res = LSTM(
        128
        , return_sequences=True
        , name='generator_LSTM_postconcat'
        , trainable=generator_trainable
        , kernel_initializer='he_normal'
        , recurrent_initializer='he_normal'
    )(res)

    res = TimeDistributed(Dropout(0.05))(res)
    conf = TimeDistributed(
        Dense(
          enc_params
          , activation="relu"
          , trainable=generator_trainable
          , kernel_initializer='he_normal'
        )
        , name='generator_TD_Dense_0'
    )
    res = generator_output = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)


    # res = self.decoder_model(framelen, (-1, enc_params))(generator_output)

    conf = self.decoder_model(framelen)
    res = conf(res)
    print("decoder_model shapes for input / output 1")
    print(conf.get_input_shape_at(1))
    print(conf.get_output_shape_at(1))


    main_output = res

    model = Model(
        #inputs=[main_input, short_input],
        inputs=[main_input],
        outputs=[main_output, mid_output, generator_output, encoder_output]
    )

    self.model = model
    return model


  def encoder_model(self, enc_params, shape=(-1,0)):


    if self.models.get('encoder_model'):
      print("Reusing encoder model")
      return self.models.get('encoder_model')
    framelen = shape[1]
    in_scale = 2
    pt_len = 3
    in_count = framelen * in_scale
    encoder_trainable = self.encoder_trainable

    conv_count = self.conv_count

    res = encoder_input = Input(shape=shape, dtype='float32', name="encoder_input")
    print("Encoder model input shape:", shape)

    res = keras.layers.concatenate([res, res])

    conf = TimeDistributed(keras.layers.Reshape((in_count, 1), trainable=encoder_trainable))
    res = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = Conv2D(
              conv_count
              , (3,14)
              , strides=(2,1)
              , padding='valid'
              , data_format='channels_last'
              , activation='relu'
              , kernel_initializer='he_normal'
              , trainable=encoder_trainable
            )
    res = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = Conv2D(
              conv_count
              , (3,13)
              , strides=(2,1)
              , padding='valid'
              , data_format='channels_last'
              , activation='relu'
              , kernel_initializer='he_normal'
              , trainable=encoder_trainable
            )
    res = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    res = keras.layers.Reshape((-1, conv_count), trainable=encoder_trainable)(res)

    # conf = TimeDistributed(
    #     Dense(
    #         conv_count
    #         , activation="softmax"
    #         , kernel_initializer='he_normal'
    #         , trainable=encoder_trainable
    #         , name="encoder_softmax"
    #     )
    # )
    # res = conf(res)


    # Second chain - pooling input data parameter by parameter
    conf = TimeDistributed(keras.layers.Reshape((framelen, 1), trainable=encoder_trainable))
    shaped_input = conf(encoder_input)

    conf = AveragePooling2D(pool_size=(3,1), strides=(2,1), padding="valid", trainable=encoder_trainable)
    res_mean = conf(shaped_input)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)
    conf = AveragePooling2D(pool_size=(3,1), strides=(2,1), padding="valid", trainable=encoder_trainable)
    res_mean = conf(res_mean)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)
    conf = keras.layers.Reshape((-1, framelen), trainable=encoder_trainable)
    res_mean = conf(res_mean)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)
    # conf = TimeDistributed(
    #     Dense(
    #         enc_params
    #         , activation="relu"
    #         , trainable=encoder_trainable
    #         , name="encoder_post_average_dense"
    #     )
    # )
    # res_mean = conf(res_mean)
    # print(conf.get_config())
    # print(conf.input_shape)
    # print(conf.output_shape)

    res = keras.layers.concatenate([res, res_mean])

    conf = TimeDistributed(
        Dense(
            enc_params * 5
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=encoder_trainable
            , name="encoder_concat_chains_dense"
        )
    )
    res = conf(res)

    conf = TimeDistributed(
        Dense(
            enc_params - pt_len
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=encoder_trainable
            , name="encoder_final_dense"
        )
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)



    conf = TimeDistributed(
        Dense(
            enc_params - pt_len
            , activation="softmax"
            , kernel_initializer='he_normal'
            , trainable=encoder_trainable
            , name="encoder_softmax"
        )
    )
    res = conf(res)




    # Add a digest of the second chain into the output
    conf = TimeDistributed(
        Dense(
            framelen * 3
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=encoder_trainable
            , name="encoder_concat_chains_dense"
        )
    )
    res_mean = conf(res_mean)

    conf = TimeDistributed(
        Dense(
            pt_len
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=encoder_trainable
            , name="encoder_concat_chains_dense"
        )
    )
    res_mean = conf(res_mean)

    # Concatenate the results
    res = keras.layers.concatenate([res, res_mean])

    encoder_output = res

    self.models['encoder_model'] = Model(encoder_input, encoder_output)
    return self.models['encoder_model']

  # Shared Decoder model, taking Conv2D encoded input, or an attempt at predicted
  # sequences of the encoded data.
  #
  # Returns a single instance of a decoder model. May be called any number of times.
  #
  # Arguments:
  #  framelen: standard frame length for a timestep
  #  shape: two element tensor, typically (-1, enc_params), where enc_params is the
  #         width of the input timestep. -1 saves us having to enter the
  #         frame sequence (sub-batch) length
  #
  # Learning can be disabled by setting self.decoder_trainable = False
  #
  # Example:
  #    main_output = self.decoder_model(framelen, (-1, enc_params))(generator_output)
  def decoder_model(self, framelen, shape=(-1, 0)):

    if self.models.get('decoder_model'):
      return self.models.get('decoder_model')

    pt_len = 3

    decoder_trainable = self.decoder_trainable
    enc_params = shape[1]
    conv_count = self.conv_count

    res = decoder_input = Input(shape=shape, dtype='float32', name="decoder_input")

    # conf = TimeDistributed(
    #     Dense(
    #         enc_params - pt_len
    #         , activation="relu"
    #         , trainable=decoder_trainable
    #         , name='decoder_pre_conv_dense'
    #     )
    # )
    conf = TimeDistributed(
      Lambda(
        lambda x: x[:, : enc_params-pt_len]
        , output_shape=(enc_params-pt_len,)
        , trainable=decoder_trainable
      )
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = TimeDistributed(
      keras.layers.Reshape(
        (1, enc_params-pt_len)
        , trainable=decoder_trainable
      )
      # , input_shape=(shape[0], enc_params-pt_len)
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = Conv2DTranspose(
              conv_count
              , kernel_size=(3,13)
              , padding='valid'
              , strides=(2,1)
              , activation='relu'
              , data_format="channels_last"
              , kernel_initializer='he_normal'
              , trainable=decoder_trainable
    )
    res = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = Conv2DTranspose(
              conv_count
              , kernel_size=(3,14)
              , strides=(2,1)
              , padding='valid'
              , activation='relu'
              , data_format="channels_last"
              , kernel_initializer='he_normal'
              , trainable=decoder_trainable
    )

    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    # Return a single filter pulling together the results of all conv_count filters
    conf = Conv2D(
              1
              , kernel_size=(2,14)
              #  , strides=(2,1)
              , padding='valid'
              , activation='sigmoid'
              , name='decoder_conv_squash'
              , kernel_initializer='he_normal'
              , trainable=decoder_trainable)
    res = conf(res)



    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf  = keras.layers.Reshape((-1, framelen), trainable=decoder_trainable)
    res = conf(res)

    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)



    # Pass through the input
    conf = TimeDistributed(
      Lambda(
        lambda x: x[:, enc_params-pt_len : enc_params]
        , output_shape=(pt_len,)
        , trainable=decoder_trainable
      )
    )



    # conf = TimeDistributed(
    #     Dense(
    #         pt_len
    #         , activation="relu"
    #         , trainable=decoder_trainable
    #         , name='decoder_pass_thru_dense_in'
    #     )
    # )
    res_pt = conf(decoder_input)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)


    conf = TimeDistributed(keras.layers.Reshape((pt_len, 1), trainable=decoder_trainable))
    res_pt = conf(res_pt)

    conf = UpSampling2D(
        size=[4,1]
        , data_format='channels_last'
        , trainable=decoder_trainable
    )
    res_pt = conf(res_pt)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = ZeroPadding2D(padding=(1,0), trainable=decoder_trainable)
    res_pt = conf(res_pt)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = keras.layers.Reshape((-1, pt_len), trainable=decoder_trainable)
    res_pt = conf(res_pt)

    conf = TimeDistributed(
        Dense(
            framelen * 5
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=decoder_trainable
            , name='decoder_pass_thru_dense_wide'
        )
    )
    res_pt = conf(res_pt)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = TimeDistributed(
        Dense(
            framelen
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=decoder_trainable
            , name='decoder_pass_thru_dense'
        )
    )
    res_pt = conf(res_pt)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)


    # Final stage of decoder
    res = keras.layers.concatenate([res, res_pt])

    conf = TimeDistributed(
        Dense(
            enc_params * 5
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=decoder_trainable
            , name='decoder_post_conv_dense'
        )
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    conf = TimeDistributed(
        Dense(
            framelen
            , activation="relu"
            , kernel_initializer='he_normal'
            , trainable=decoder_trainable
            , name='decoder_final_dense'
        )
    )
    res = conf(res)
    print(conf.get_config())
    print(conf.input_shape)
    print(conf.output_shape)

    decoder_output = res

    self.models['decoder_model'] = Model(decoder_input, decoder_output)
    return self.models['decoder_model']


  def loss_types(self):
    loss = CustomObjects.codec2_param_error_td
    # other loss options: CustomObjects.codec2_param_mean_square_error; 'mean_absolute_error'; 'cosine_proximity'

    return [loss, loss, 'mean_absolute_error', 'mean_absolute_error']

  def compile_model(self):
    self.utils.log("Compiling model")


    encoder_loss_prop = 0

    main_loss_prop = 0.0
    mid_loss_prop = 0.05
    generator_loss_prop = 0.95

    if self.decoder_trainable and not self.generator_trainable:
      main_loss_prop = 0
      mid_loss_prop = 1
      generator_loss_prop = 0
    elif not self.decoder_trainable and self.generator_trainable:
      mid_loss_prop = 0
      main_loss_prop = 0.0
      generator_loss_prop = 1.0

    self.utils.log("Loss weightings:", main_loss_prop, mid_loss_prop, generator_loss_prop, encoder_loss_prop)

    self.model.compile(
        loss=self.loss_types(),
        loss_weights=[main_loss_prop, mid_loss_prop, generator_loss_prop, encoder_loss_prop],
        optimizer=self.get_optimizer_from_config())
    self.utils.log_model_summary()

  def fit(self, input_seq, output_seq, batch_size=None, epochs=1, shuffle=False, callbacks=None):
      inputs = input_seq[0]
      outputs = output_seq

      # s = output_seq[0].shape
      # dummy_encoded_output = np.zeros((s[0], 24, 64), dtype=np.float32)
      # outputs.append( dummy_encoded_output)

      print("X shape:", inputs.shape)
      print("y shape:", outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape)

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
