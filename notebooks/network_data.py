import json
from pprint import pprint
import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import os

home = os.environ.get('HOME')

def model_config(network_tag):
  with open(home + "/store/c2gen/out/" + network_tag + "/jmodel-0.json") as data_file:    
    data = json.load(data_file)
  
  #pprint(data)
  config = data['config']
  for cs in config:
    c = cs['config']
    print(c['name'], ":", cs['class_name'])
    print("trainable? ", c['trainable'])
    print(c.get('units',""), "units")
    print(c.get('activation', ""), "activation")
    print(c.get('dropout',""), "dropout")
    print()
    
def plot_training_loss(network_tag):
  dataframe = pandas.io.parsers.read_csv(home + "/store/c2gen/out/" + network_tag + "/training.log")
  data = dataframe.as_matrix(None)
  plt.plot(data)
  plt.xlabel('iteration')
  plt.ylabel('loss (mean absolute)')
  plt.title('Training Loss for\nnetwork: ' + network_tag)
  plt.grid(True)
#  fn = "loss-plot-" + network_tag + ".png"
#  plt.savefig(fn)
  plt.show()




def plot_codec_params(network_tag, iteration):
  iteration = str(iteration)
  infilename = home + "/store/c2gen/out/" + network_tag + "/out-c2cb-" + iteration
  indata = np.fromfile(infilename, dtype=np.uint8)
  
  data = np.reshape(indata, (-1,16))
  data = np.multiply(data, [16,16,16,16,1,1,4,4,4,4,4,4,4,4,4,4])
  
  plt.plot(data)
  plt.xlabel('time (frames)')
  plt.ylabel('audio params (units)')
  plt.title('Codec Params\n' + network_tag + " @ iteration " + iteration)
  plt.grid(True)
#  fn = "codec-plot-" + network_tag + "-" + iteration + ".png"
#  plt.savefig(fn)
  plt.show()

def plot_audio_waveform(network_tag, iteration):
  iteration = str(iteration)
  infilename = home + "/store/c2gen/out/" + network_tag + "/out-c2cb-" + iteration + ".raw"
  data = np.fromfile(infilename, dtype=np.int16)
  
  plt.plot(data)
  plt.xlabel('time (samples)')
  plt.ylabel('audio waveform')
  plt.title('Audio Waveform\n' + network_tag + " @ iteration " + iteration)
  plt.grid(True)
#  fn = "audio-plot-" + network_tag + "-" + iteration + ".png"
#  plt.savefig(fn)
  plt.show()

def plot_gen_audio_waveform(infilename):

  infilename = home + "/store/c2gen/generated/"+infilename+".wav"
  data = np.fromfile(infilename, dtype=np.int16)
  
  plt.plot(data)
  plt.xlabel('time (samples)')
  plt.ylabel('audio waveform')
  plt.title('Audio Waveform\n')
  plt.grid(True)
#  fn = "audio-plot-" + network_tag + "-" + iteration + ".png"
#  plt.savefig(fn)
  plt.show()