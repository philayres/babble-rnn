import sys
import os

from keras.models import load_model
from custom_objects import CustomObjects
from model_def import ModelDef
  


Train=1
Generate=2

class ModelUtils(object):

  
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
  model_def = None
  generate_len = 200
  
  def __init__(self):
  
    if len(sys.argv) < 3:
      print("training usage: lstm_c2_generation <tagname> <test data filename> [load model filename]")
      print("for example\n lstm_c2_generation test1 test/LDC97S44-8k.c2cb")
      print("generator usage: lstm_c2_generation --generate=<base filename> [--seed_index=<'random'|frame num|time in seconds>] [--generate-len=<frames>] <test data filename> <load model filename>")
      print("for example\n lstm_c2_generation --generate=audiofile --seed_index=60s --generate-len=500 test/LDC97S44-8k.c2cb out/realmodel/model-600.h5")
      exit()
    
    named_args = {}
    basic_args = []
    
    self.named_args = named_args
    self.basic_args = basic_args
    for i, arg in enumerate(sys.argv[1:]):
      if arg[0:2] == "--":
        a = arg.split("=")
        key = a[0][2:]
        named_args[key] = a[1]
      else:
        basic_args.append(arg)  
    
    if named_args.get('generate', None):
      self.model_tag = named_args['generate']
      self.mode = Generate
      basic_args.insert(0, None)
    else:
      self.model_tag = basic_args[0]
      self.mode = Train
    
      
    self.testdata_filename = basic_args[1]
    if len(basic_args) > 2:
      self.model_filename = basic_args[2]  
    
    if self.training_mode():
      self.output_dir="out/"+str(self.model_tag)+"/"
      self.output_fn=self.output_dir+"out-c2cb-"    
      try:
        os.makedirs(self.output_dir)
      except OSError:
        print("the tag ", self.model_tag, " has been used")
        exit()  
    else:
      self.output_dir="generated/"
      self.output_fn=self.output_dir+str(self.model_tag)
      
    if named_args.get('generate-len', None):
      self.generate_len = int(named_args['generate-len'])
          
    self.h5_model_filename=self.output_dir+"model-"
    self.h5_weights_filename=self.output_dir+"weights-"
  
    from keras.callbacks import CSVLogger
    self.csv_logger_fn = self.output_dir + 'training.log'
    self.csv_logger = CSVLogger(self.csv_logger_fn, append=True)
    self.logfile_fn = self.output_dir + "log"
    self.logfile = open(self.logfile_fn, "w")
    
    
  def  setup_seed_start(self, generator):
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
    return model.save(self.h5_model_filename+str(iteration)+".h5")

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
    
  def log(self, *inargs):
    
    for arg in inargs: 
      self.logfile.write(str(arg)+" ")
      print(str(arg)),
    print
    self.logfile.write("\n")
    self.logfile.flush()
 
  def signal_handler(self, signal, frame): 
    self.log('Interrupt signal caught. Closing gracefully.') 
    self.logfile.close()
    sys.exit(0)
  
  def custom_objects(self):
    return {"CustomObjects": CustomObjects, 
      "codec2_param_error":CustomObjects.codec2_param_error}

  def test_seed_data(all_frames, start_index):
    self.open_output_file(0)
    seed_frame_seq = all_frames[start_index: start_index + frame_seq_len]

    for frame in seed_frame_seq:
      self.output_file.write(sample(frame))

    self.output_file.close()

  def define_or_load_model(self, frame_seq_len, framelen, num_frame_seqs):
    self.model_def = ModelDef(self)
    if len(self.model_filename) > 0:
      model = self.load_model()   
      self.save_json_model()
    else:
      self.log("creating new model")
      model = self.model_def.define_model(frame_seq_len, framelen, num_frame_seqs)
      self.save_json_model()
    
    return self.model_def
    

  def training_mode(self):
      return self.mode == Train
      
  def generate_mode(self):
      return self.mode == Generate
      