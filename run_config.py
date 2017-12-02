import os
import json
from model_def import ModelDef
from custom_objects import CustomObjects

class RunConfig(object):

  # start iteration to use if the iteration counter file is empty
  start_iteration = 0

  # number of training iterations
  num_iterations = 1200

  # number of frame sequences (each of length frame_seq_len) that make up a batch during .fit()
  # gradients are updated after each of these batches
  # some research suggests reducing the size of batches to increase generalisation during learning,
  # although at the expense of slower training
  # Note: for TimeDensity (and maybe stateful) learning, the fit_batch_size acts more like
  # a counter of the number of complete frame sequences. Therefore if you retain a high number
  # there are very few updates happening (one every frame_seq_len * fit_batch_size)
  # and learning stalls
  fit_batch_size = 100

  # learn and generate with just a single timestep (True) or
  # use a block (and a TimeDistributed output)
  learn_next_step = True

  # generate sample data every nth iteration
  gen_every_nth = 30

  # Generate just the main output (1) or mid output too (2)
  generate_num_outputs = 2

  # save model every nth iteration
  save_model_every_nth = 30

  # number of bytes (unsigned 8 bit) in a Codec 2 frame
  # for 1300 rate codec one frame encodes 40ms of raw PCM audio
  # framelen=16
  # for 300 rate codec one frame encodes 20ms of raw PCM audio
  framelen = 13

  # time length of a frame in ms
  frame_len_ms = 20

  # length of frame sequence for learning
  frame_seq_len = 100

  # overlap the sequence at each end and provide a secondary non-overlapped version as an input
  # Disable by setting to 0
  overlap_sequence = 2

  # the seed sequence length is the number of frames the generator expects to be input
  # as the seed. This must match the frame_seq_len currently
  seed_seq_len = frame_seq_len

  # number of frames between the start of each sequence of frames used during learning in a batch
  # if this equals frame_seq_len, then the frame sequences will be contiguous and will not overlap
  # picking a number less than frame_seq_len provides overlapping frames every seq_step to add to the training set
  # picking a number larger than frame_seq_len means that frames from the corpus will be skipped
  # note that overlapping frame sequences may be considered a way to augment data, but also increases memory requirements,
  # since a single batch of data will actually be larger than the original corpus
  seq_step = frame_seq_len - overlap_sequence*2

  # filename including relative path to the test data
  # this filename may be overridden by setting on the command line, in which case this setting will be updated to match
  test_data_fn = None

  # number of frames generated after the seed when generating new data during training iterations
  # when using --generate=audiofile on the command line, this setting is ignored
  generate_len = 200

  # flag to indicate use of Stateful LSTMs. shuffle should be set to False  when using stateful=True
  # learn_next_step = False is probably also required
  stateful = False

  # shuffle frame sequences within each batch. Stateful LSTMs should not receive shuffled data if they
  # are to learn long time sequences successfully.
  # Some evidence suggests that not shuffling time series data even in non-stateful operation can aid
  # learning, although this is not well recognised.
  shuffle = False

  # in order to handle a large corpus, each iteration the fit function can be passed the next subset of corpus
  # data. This enables the whole batch to fit into the GPU, avoiding out of memory issues.
  # The limit_frames setting is the maximum number of *frame sequences* to be used in a batch each iteration.
  # Note that the final subset will be unused if it does not match exactly the limit_frames size
  limit_frames = 0 #int(4845288/2)

  # filename of the model. This will be overridden by the command line, and updated appropriately.
  # During training, the model_filename setting will be updated when a new model .h5 file is successfully saved
  model_filename = ""

  # the Keras optimizer to be used
  # name: exact case-sensitive name to match the class of the optimizer
  # params: the optional parameters to be used by the optimizer, e.g. {"lr": 0.01,...}
  optimizer = {
    "name": "Nadam",
    "params": {}
  }

  # the scaling factors used for normalising frame parameters to range 0..1
  frame_prop_orig_scale = [
   1,
   127,
   31 ,
   31,31,31,31,31,31,31,31,31,31
  ]

  # allow different weightings for loss calculation
  frame_prop_loss_scale = [
     1000.0,
     1000.0,
     1000.0,
     700.0,600.0,500.0,400.0,300.0,200.0,100.0,60.0,31.0,16.0
  ]


  ##### No more user config variables to change #####

  # the attributes to be loaded and saved to the config.json file
  config_attrs = [
    "start_iteration",
    "num_iterations",
    "fit_batch_size",
    "learn_next_step",
    "gen_every_nth",
    "save_model_every_nth",
    "framelen",
    "frame_seq_len",
    "overlap_sequence",
    "seed_seq_len",
    "seq_step",
    "test_data_fn",
    "frame_prop_orig_scale",
    "frame_prop_loss_scale",
    "stateful",
    "shuffle",
    "limit_frames",
    "optimizer",
    "generate_len",
    "model_filename",
    "frame_len_ms"
  ]

  # Unsaved items and state
  utils = None
  config_json_fn = "config.json"
  num_frames = None

  def __init__(self, utils):
    self.utils = utils or self

    assert (self.utils.output_dir != None and self.utils.output_dir != '')

    if self.utils:
      self.config_json_fn  = self.utils.output_dir + self.config_json_fn
      self.load_config()
    return

  def log(self, x):
    print("Not Logged: ", x)

  def load_config(self):
    res = None
    if os.path.isfile(self.config_json_fn):
      with open(self.config_json_fn) as f:
        res = json.load(f)
        self.utils.log("Using JSON configuration from: " + self.config_json_fn)
    else:
      self.utils.log("Using default config")

    if res:
      for k, v in res.items():
        setattr(self, k, v)
    self.after_load()
    return res

  def after_load(self):
    CustomObjects.frame_prop_loss_scale = self.frame_prop_loss_scale
    if self.stateful != None:
      ModelDef.stateful = self.stateful

  def save_config(self):
    self.utils.log("saving config")
    with open(self.config_json_fn, "w") as f:
      res = {}
      for a in self.config_attrs:
        res[a] = getattr(self, a)

      json.dump(res, f)

  def log_attrs(self):
    for a in self.config_attrs:
      self.utils.log(a + ": " + str(getattr(self, a)))
