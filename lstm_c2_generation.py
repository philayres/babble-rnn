from __future__ import print_function

from model_utils import ModelUtils
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import time
import sys
import os
import signal
from generator import Generator

utils = ModelUtils()

  
signal.signal(signal.SIGINT, utils.signal_handler)
signal.signal(signal.SIGTERM, utils.signal_handler)
# number of training iterations
num_iterations = 600
start_iteration = 1

fit_batch_size = 200 #128
utils.log("fit_batch_size: ", fit_batch_size)
# length of frame sequence to generate
genlen=400
utils.log("genlen: ", genlen)
# generate sample data every nth iteration
gen_every_nth = 5

# number of bytes (unsigned 8 bit) in a Codec 2 frame
# note: one frame encodes 40ms of raw PCM audio
framelen=16

'''
frame_property_bits = [
 1,1,1,1, # voiced flags for 4 (10ms) PCM frames
 7, #Wo
 5, #E
 4,4,4,4,4,4,4,3,3,2 #LSP
]
'''

# length of frame sequence for learning
frame_seq_len = 200 # 8 seconds of audio
seed_seq_len = frame_seq_len
utils.log("frame_seq_len: ", frame_seq_len)

seq_step = int(frame_seq_len/1.2) #int(frame_seq_len/10)
utils.log("seq_step: ", seq_step)

model_def = None

utils.log("loading test data from: ", utils.testdata_filename)
testdata = np.fromfile(utils.testdata_filename, dtype=np.uint8)

len_testdata = len(testdata)
num_frames = int(len_testdata / framelen)
utils.log('corpus length:', len_testdata)


# cut the testdata into sequences of frame_seq_len characters
print("scanning testdata into frames and frame sequences")

frame_seqs = []
next_frames = []
all_frames = []


####  Setup the model
model_def = utils.define_or_load_model(frame_seq_len, framelen)


def normalize_input(frame):
  normframe = np.array(frame, dtype=np.float32)
  normframe = np.divide(normframe, model_def.frame_property_scaleup)
  return normframe

def gen_sequence(iteration):
  return (iteration % gen_every_nth == 0)


# step through the testdata, pulling those bytes into an array of all the the frames, all_frames
for j in range(0, num_frames):
    i = j * framelen   
    all_frames.append(normalize_input(testdata[i: i + framelen]))

utils.log('number of frames:', len(all_frames))

# pull the frames into frame sequences (frame_seqs), each of frame_seq_len frames
# pull a single frame following each frame sequence into an array of next_frames
for i in range(0, num_frames - frame_seq_len, seq_step):
    frame_seqs.append(all_frames[i: i + frame_seq_len])
    next_frames.append(all_frames[i + frame_seq_len])



utils.log('number of frame sequences:', len(frame_seqs))


print('initialising input and expected output arrays')
X = np.zeros((len(frame_seqs), frame_seq_len, framelen), dtype=np.float32)
y = np.zeros((len(frame_seqs), framelen), dtype=np.float32)


for i, frame_seq in enumerate(frame_seqs):
    # expected output is always the next frame for corresponding frame_seq
    y[i] = next_frames[i]

    # input is just each frame_seq 
    X[i] = frame_seq



generator = Generator(utils, all_frames, seed_seq_len, genlen)
generator.frame_property_scaleup = model_def.frame_property_scaleup
generator.framelen = framelen
utils.setup_seed_start(generator)

if utils.generate_mode():
  utils.log("Generating Samples")
  generator.generate(0)
  
  exit()

# train the model
# output generated frames after nth iteration
for iteration in range(start_iteration, num_iterations + 1):
  print('-' * 50)
  
  
  utils.log('Training Iteration', iteration)
  
  model_def.before_iteration(iteration)
    

  model_def.model.fit(X, y, batch_size=fit_batch_size, nb_epoch=1,
   callbacks=[utils.csv_logger]
  )

  if gen_sequence(iteration):
    # every nth iteration generate sample data as a Codec 2 file

    utils.log("Generating samples")
    generator.generate(iteration)
    
    print("saving .h5 model file")
    utils.save_h5_model(iteration)
    print("saving .h5 weights file")      
    utils.save_weights(iteration)
    
  else:
    print("not generating samples this iteration")  

  print()


