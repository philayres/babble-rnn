import os
import json
from model_def import ModelDef
from custom_objects import CustomObjects

class RunConfig(object):

  # start iteration to use if the iteration counter file is empty
  start_iteration = 0
  
  # number of training iterations
  num_iterations = 1200
  
  # number of frames to use during learning with .fit()
  fit_batch_size = 200
  
  # learn and generate with just a single timestep (True) or 
  # use a block (and a TimeDistributed output)
  learn_next_step = True
  
  # generate sample data every nth iteration
  gen_every_nth = 20
  
  # save model every nth iteration
  save_model_every_nth = 10 
  
  # number of bytes (unsigned 8 bit) in a Codec 2 frame
  # for 1300 rate codec one frame encodes 40ms of raw PCM audio
  # framelen=16
  # for 300 rate codec one frame encodes 20ms of raw PCM audio
  framelen = 13
  # length of frame sequence for learning
  frame_seq_len = 100 # 2 seconds of audio for 3200 codec 
  #frame_seq_len = 100 # 4 seconds of audio for 1300 codec  
  
  # pick overlapping frames every seq_step to add to the training set 
  seed_seq_len = 100 
  
  seq_step = 80
  test_data_fn = None  
  
  
  stateful = True
  shuffle = not stateful
  
  limit_frames = int(4845288/2)
  
  optimizer = {
    "name": "RMSprop",
    "params": {}
  }
  
  frame_prop_loss_scale = [
   1,
   2**7,
   2**5,
   32,32,32,32,32,32,32,32,32,32
  ]
  
  
  config_attrs = [
    "start_iteration",
    "num_iterations",
    "fit_batch_size",
    "learn_next_step",
    "gen_every_nth",
    "save_model_every_nth",
    "framelen",
    "frame_seq_len",
    "seed_seq_len",
    "seq_step",
    "test_data_fn",
    "frame_prop_loss_scale",
    "stateful",
    "shuffle",
    "limit_frames",
    "optimizer"
    
  ]
  
  utils = None
  config_json_fn = "config.json"
  
  def __init__(self, utils):
    self.utils = utils or self
    if self.utils:
      self.config_json_fn  = self.utils.output_dir + self.config_json_fn
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
    
    with open(self.config_json_fn, "w") as f:
      res = {}
      for a in self.config_attrs:
        res[a] = getattr(self, a)
        
      json.dump(res, f)
  
  def log_attrs(self):
    for a in self.config_attrs:
      self.utils.log(a + ": " + str(getattr(self, a)))
      
      