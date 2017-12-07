import sys
import os

from keras.models import load_model
from keras.utils import plot_model
from custom_objects import CustomObjects
from model_def import ModelDef
from run_config import RunConfig
import datetime

Train=1
Generate=2

class ModelUtils(object):

  iteration = 0
  config = None
  mode = Train
  model_filename = ""
  model_tag = ""
  testdata_filename = ""
  output_dir = ""
  h5_model_filename = ""
  h5_weights_filename = ""
  output_fn = ""
  output_file = None
  csv_logger_fn = ""
  csv_logger = None
  logfile_fn = ""
  logfile = None
  iteration_counter_fn = None
  model_def = None
  one_off_generate_len = None
  load_weights = None

  def __init__(self):
    self.buffered_logs = []
    self.log("====================================================")
    self.log("Started New Run at:",  datetime.datetime.now())
    self.log("PID:", os.getpid())
    self.log("====================================================")


    if len(sys.argv) < 2:
      print("training usage: lstm_c2_generation <tagname> [test data filename>] [load model filename]")
      print("for example\n lstm_c2_generation test1 test/LDC97S44-8k.c2cb")
      print("if test data filename or load model filename are excluded, the settings in config.json will be used if it exists")
      print("if load model filename is set to 'none' then the coded model definition will be used, regardless of what is set in the config.json file.\n")
      print("generator usage: lstm_c2_generation [tagname] --generate=<base filename> [--seed_index=<'random'|frame num|time in seconds>] [--generate-len=<frames>] <test data filename> <load model filename>")
      PRINT("loading weights: lstm_c2_generation [tagname] --load-weights=<path to .h5 file> <test data filename> <load model filename>")
      print("for example\n lstm_c2_generation --generate=audiofile --seed_index=60s --generate-len=500 test/LDC97S44-8k.c2cb out/realmodel/model-600.h5")
      exit()

    named_args = {}
    basic_args = []

    self.named_args = named_args
    self.basic_args = basic_args

    print('arguments:', sys.argv)
    exit()
    for i, arg in enumerate(sys.argv[1:]):

      self.log('arg', i, arg)
      print('arg', 1, arg)
      if arg[0:2] == "--":
        a = arg.split("=")
        key = a[0][2:]
        named_args[key] = a[1]
      else:
        basic_args.append(arg)

    if named_args.get('generate', None):
      self.generate_name = named_args['generate']
      self.mode = Generate
      self.model_tag = basic_args[0]
      self.log("mode: Generate")
    else:
      self.model_tag = basic_args[0]
      self.mode = Train
      self.log("mode: Train")

    if self.training_mode():
      self.output_dir="out/"+str(self.model_tag)+"/"
      self.output_fn=self.output_dir+"out-c2cb-"
      try:
        os.makedirs(self.output_dir)
      except OSError:
        print("the tag ", self.model_tag, " has been used")
        print("continuing where we left off")
    else:
      self.output_dir="out/"+str(self.model_tag)+"/"
      self.output_fn="generated/"+str(self.generate_name)

    exit()


    self.config = RunConfig(self)

    if len(basic_args) > 1:
      self.testdata_filename = basic_args[1]
      self.config.test_data_fn = self.testdata_filename
      self.log("using command line test data filename:", self.config.test_data_fn)


    if len(basic_args) > 2:
      self.model_filename = basic_args[2]
      self.log("using command line model filename:",self.model_filename)
    else:
      self.model_filename = self.config.model_filename
      self.log("using configured model_filename:",self.config.model_filename)



    if named_args.get('generate-len', None):
      self.one_off_generate_len = int(named_args['generate-len'])

    if named_args.get('load-weights', None):
      self.load_weights = named_args['load-weights']
      self.log("loading weights from a weights file:", self.load_weights)
      self.model_filename = 'none'
    else:
      self.log("not loading weights from a weights file")



    self.h5_model_filename=self.output_dir+"model-"
    self.h5_weights_filename=self.output_dir+"weights-"

    if self.training_mode():
      from keras.callbacks import CSVLogger
      self.csv_logger_fn = self.output_dir + 'training.log'
      self.csv_logger = CSVLogger(self.csv_logger_fn, append=True)
      self.iteration_counter_fn = self.output_dir + "iteration_counter"
      self.gen_counter_fn = self.output_dir + "gen_counter"
    self.logfile_fn = self.output_dir + "log"
    self.logfile = open(self.logfile_fn, "a", 1)




  def setup_seed_start(self, generator):
    if self.named_args.get('seed_index', None):
      seed_index = self.named_args['seed_index']
      if seed_index == 'random':
        self.log("Setting seed start index to 'random'")
        generator.set_random_seed_start_index()
      elif seed_index.find('s') > 0:
        self.log("Setting seed start index to:", seed_index)
        seed_index = seed_index[0:-1]
        generator.set_time_seed_start_index(int(seed_index))
      else:
        self.log("Setting seed start index to:", seed_index)
        generator.set_frame_seed_start_index(int(seed_index))

  def load_model(self):

    self.log("loading model: " + self.model_filename)
    self.model_def.model = load_model(self.model_filename, custom_objects=self.custom_objects())

    if self.training_mode():
      self.log("saving config after loading model")
      self.config.model_filename = self.model_filename
      self.config.save_config()
    else:
      self.log("not saving config after loading model")
    self.log_model_summary()
    return self.model_def.model

  def save_json_model(self, update_num=0):
    model = self.model_def.model
    json_string = model.to_json()
    print("saving json model")
    n = "jmodel-"+str(update_num)+".json"
    mfile= open(self.output_dir + n, "w")
    mfile.write(json_string)
    mfile.close

  def save_h5_model(self, iteration):
    model = self.model_def.model
    fn = self.h5_model_filename+str(iteration)+".h5"
    res = model.save(fn)


    self.config.model_filename = fn
    self.write_iteration_count(iteration)
    self.config.save_config()

    #plot_model(model, to_file=self.output_dir+'vis-model-'+str(iteration)+'.png')

    return res

  def save_weights(self, iteration):
    model = self.model_def.model
    return model.save_weights(self.h5_weights_filename+str(iteration)+".h5")

  def open_output_file(self, iteration):
    if self.training_mode():
      output_fn = self.output_fn+str(iteration)
    else:
      output_fn = self.output_fn
    self.output_file = open(output_fn, "wb")
    return output_fn

  def after_iteration(self, iteration):
    #self.write_iteration_count(self, iteration)
    return

  def write_iteration_count(self, iteration):
    self.config.start_iteration = iteration
    with open(self.iteration_counter_fn, "w") as f:
      f.write(str(iteration))

  def write_gen_count(self, iteration):
    with open(self.gen_counter_fn, "w") as f:
      f.write(str(iteration))

  def read_iteration_count(self):
    res = []
    if self.iteration_counter_fn and os.path.isfile(self.iteration_counter_fn):
      with open(self.iteration_counter_fn) as f:
        res = f.readlines()

    if len(res) == 1:
      i = int(res[0])
      self.iteration = i
      self.log("Continuing from a previous run at iteration: ", i)
      return i
    else:
      self.iteration = 0
      self.log("No iteration file found. Setting to 0.")
      return 0

  def log(self, *inargs):

    if self.logfile == None:
      args = []
      for a in inargs:
        args.append(str(a))
      self.buffered_logs.append(str.join(" ", args) + "\n")
      return
    elif len(self.buffered_logs) > 0:
      for s in self.buffered_logs:
        print(s)
        self.logfile.write(s)
        self.logfile.flush()
        self.buffered_logs = []

    try:
      for arg in inargs:
        self.logfile.write(str(arg)+" ")
        print(str(arg)),
      print

      self.logfile.write("\n")
      self.logfile.flush()
    except IOError:
      print("* Logging Failed *")
      for arg in inargs:
        print(str(arg)),
      print


  def signal_handler(self, signal, frame):

    self.log('Interrupt signal caught. Closing gracefully.')
    self.write_iteration_count(self.iteration)

    print("saving .h5 model file")
    self.save_h5_model(self.iteration)
    print("saving .h5 weights file")
    self.save_weights(self.iteration)
    print("exiting now")
    self.logfile.close()
    sys.exit(0)

  def custom_objects(self):
    return {"CustomObjects": CustomObjects,
      "codec2_param_error":CustomObjects.codec2_param_error,
      "codec2_param_error_td":CustomObjects.codec2_param_error}

  def test_seed_data(self, all_frames, start_index):
    self.open_output_file(0)
    seed_frame_seq = all_frames[start_index: start_index + frame_seq_len]

    for frame in seed_frame_seq:
      self.output_file.write(sample(frame))

    self.output_file.close()

  def define_or_load_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.model_def = ModelDef(self, self.config)

    if len(self.model_filename) > 0 and self.model_filename != 'none' and self.model_filename != 'None':
      model = self.load_model()
      self.save_json_model()
    else:
      self.log("creating new model")
      model = self.model_def.define_model(frame_seq_len, framelen, num_frame_seqs)
      self.save_json_model()

    if self.load_weights != None:
      self.model_def.load_weights(self.load_weights, by_name=True)

    return self.model_def


  def training_mode(self):
      return self.mode == Train

  def generate_mode(self):
      return self.mode == Generate

  def setup_config(self):
    return self.config

  def log_model_summary(self):
    self.model_def.model.summary()
