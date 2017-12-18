codec3200 = True
codec1300 = False

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
    print("stateful", c.get('stateful',""))
    print()

def plot_training_loss(network_tag, ln, legend=None, weights=None, yscale=None, columns=None, start_index=0):
  dataframe = pandas.io.parsers.read_csv(home + "/store/c2gen/out/" + network_tag + "/training.log")

  if columns is not None:
    dataframe = dataframe[columns]

  print("Columns:", list(dataframe.columns.values))
  data = dataframe.as_matrix(None)


  if weights is not None:
    data = data * weights

  plt.plot(data[start_index : ])

  legend = legend or list(dataframe.columns.values)

  plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
  plt.xlabel('iteration')
  plt.ylabel('loss ('+ln+')')
  if yscale == 'log':
    plt.yscale('log')
  plt.title('Training Loss for\nnetwork: ' + network_tag)
  plt.grid(True)
#  fn = "loss-plot-" + network_tag + ".png"
#  plt.savefig(fn)
  plt.show()




def plot_codec_params(network_tag, iteration, scale_up='full', loc='out'):
  iteration = str(iteration)

  if loc=='out':
    fn = "/out-c2cb-" + iteration
  else:
    fn = ""

  infilename = home + "/store/c2gen/"+loc+"/" + network_tag + fn
  indata = np.fromfile(infilename, dtype=np.uint8)

  if codec1300:
    data = np.reshape(indata, (-1,16))
    if scale_up == 'full':
      data = np.divide(data, [
          1,1,1,1,
          2**7,
          2**5,
          16,16,16,16,16,16,16,8,8,4
         ])
    elif scale_up == 'orig':
      data = np.multiply(data, [16,16,16,16,1,1,4,4,4,4,4,4,4,4,4,4])
  elif codec3200:
    print("3200 rate codec\n")
    data = np.reshape(indata, (-1,13))
    if scale_up == 'full':
      data = np.divide(data, [
          1,
            2**7,
            2**5,
            32,32,32,32,32,32,32,32,32,32
         ])
    elif scale_up == 'orig':
      data

  plt.plot(data)
  plt.xlabel('time (frames)')
  plt.ylabel('audio params (units)')
  plt.title('Codec Params\n' + network_tag + " @ iteration " + iteration)
  plt.grid(True)
  plt.show()

def plot_spec_params(network_tag, iteration, params='Voicing', loc='out'):
  iteration = str(iteration)
  if loc=='out':
    fn = "/out-c2cb-" + iteration
  else:
    fn = ""

  infilename = home + "/store/c2gen/"+loc+"/" + network_tag + fn
  indata = np.fromfile(infilename, dtype=np.uint8)

  if codec1300:
    data = np.reshape(indata, (-1,16))
    if params == 'Voicing':
      data = data[:, 0:4]
    elif params == 'Wo':
      data = data[:, 4]
    elif params == 'E':
      data = data[:, 5]
    elif params == 'LSPs':
      data = data[:, 6:]
  elif codec3200:
    data = np.reshape(indata, (-1,13))
    if params == 'Voicing':
      data = data[:, 0]
    elif params == 'Wo':
      data = data[:, 1]
    elif params == 'E':
      data = data[:, 2]
    elif params == 'LSPs':
      data = data[:, 3:]

  plt.plot(data)
  plt.xlabel('time (frames)')
  plt.ylabel(params + ' (units)')
  plt.title(params + '\n' + network_tag + " @ iteration " + iteration)
  plt.grid(True)
  plt.show()


def plot_audio_waveform(network_tag, iteration):
  iteration = str(iteration)
  if codec1300:
    infilename = home + "/store/c2gen/out/" + network_tag + "/out-c2cb-" + iteration + ".raw"
  elif codec3200:
    infilename = home + "/store/c2gen/out/" + network_tag + "/out-c2cb-" + iteration + "-3200.raw"
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
