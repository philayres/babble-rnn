from __future__ import print_function

from model_utils import ModelUtils
from model_def import ModelDef
from run_config import RunConfig
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import time
import sys
import os
import signal
import math
from generator import Generator

utils = ModelUtils()
model_def = None

config = utils.setup_config()


signal.signal(signal.SIGINT, utils.signal_handler)
signal.signal(signal.SIGTERM, utils.signal_handler)

# read an existing iteration counter if it exists
start_iteration = utils.read_iteration_count() or config.start_iteration
config.start_iteration = start_iteration

num_iterations = config.num_iterations
fit_batch_size = config.fit_batch_size
if config.stateful:
  learn_next_step = False
else:
  learn_next_step = config.learn_next_step
gen_every_nth = config.gen_every_nth
save_model_every_nth = config.save_model_every_nth
framelen=config.framelen
frame_seq_len = config.frame_seq_len
seed_seq_len = config.seed_seq_len

seq_step = config.seq_step or frame_seq_len
config.seq_step = seq_step

test_data_fn = config.test_data_fn

utils.log("loading test data from: ", test_data_fn)
testdata = np.fromfile(test_data_fn, dtype=np.uint8)

len_testdata = len(testdata)
num_frames = int(len_testdata / framelen)
utils.log('corpus length (bytes):', len_testdata)
utils.log('corpus length (frames):', num_frames)
config.num_frames = num_frames

limit_frames = int(config.limit_frames)

config.log_attrs()
if not utils.generate_mode():
  config.save_config()

overlap_sequence = config.overlap_sequence

frame_seqs = []
next_frame_seqs = []
next_frames = []
current_frames = []
all_frames = []




def normalize_input(frame):
  normframe = np.array(frame, dtype=np.float32)
  normframe = np.divide(normframe, config.frame_prop_orig_scale)
  return normframe

def gen_sequence(iteration):
  return (iteration > 0) and (iteration % gen_every_nth == 0)

def save_model(iteration):
  return (iteration % save_model_every_nth == 0)

utils.log("scanning testdata into frames and frame sequences")



# step through the testdata, pulling those bytes into an array of all the the frames, all_frames
for j in range(0, num_frames):
    i = j * framelen
    all_frames.append(normalize_input(testdata[i: i + framelen]))

utils.log('actual number of frames:', len(all_frames))

if utils.generate_mode():
  num_frame_seqs = seed_seq_len
else:

  # pull the frames into frame sequences (frame_seqs), each of frame_seq_len frames
  for i in range(0, num_frames - 2*frame_seq_len, seq_step):
    frame_seqs.append(all_frames[i: i + frame_seq_len])
    if learn_next_step:
        # pull a single frame following each frame sequence into a corresponding array of next_frames
        next_frames.append(all_frames[i + frame_seq_len])
        current_frames.append(all_frames[i + frame_seq_len - 1 ])
    else:
        j = i + frame_seq_len
        next_frame_seqs.append(all_frames[(j) : (j + frame_seq_len)])

  if config.stateful and (len(frame_seqs) % fit_batch_size > 0):
    excess_frameseqs = len(frame_seqs) % fit_batch_size
    print("Stateful operation. Reducing frame sequences by:", excess_frameseqs)
    for i in range(excess_frameseqs):
      frame_seqs.pop(-1)

  utils.log('number of frame sequences:', len(frame_seqs))


  # make sure that the input and output frames are float32, rather than
  # the unsigned bytes that we load from the corpus
  print('initialising input and expected output arrays')
  num_frame_seqs = len(frame_seqs)
  X = np.zeros((num_frame_seqs, frame_seq_len, framelen), dtype=np.float32)

  # if overlap_sequence != 0:
  X2 = np.zeros((num_frame_seqs, (frame_seq_len - overlap_sequence*2), framelen), dtype=np.float32)

  if learn_next_step:
      y = np.zeros((num_frame_seqs, framelen), dtype=np.float32)
      y2 = np.zeros((num_frame_seqs, framelen), dtype=np.float32)
  else:
      y = np.zeros((num_frame_seqs, frame_seq_len - overlap_sequence*2, framelen), dtype=np.float32)
      y2 = np.zeros((num_frame_seqs, frame_seq_len - overlap_sequence*2, framelen), dtype=np.float32)



  for i, frame_seq in enumerate(frame_seqs):
      if learn_next_step:
          # expected output is always the next frame for corresponding frame_seq
          y[i] = next_frames[i]
          # The decoder output is always the current frame
          y2[i] = current_frames[i]
      else:
          y[i] = next_frame_seqs[i]
          y2[i] = frame_seq

      # input is just each frame_seq
      X[i] = frame_seq
      if overlap_sequence != 0:
          X2[i] = frame_seq[0:frame_seq_len - (2*overlap_sequence)]
      else:
          X2[i] = frame_seq

####  Setup the model
model_def = utils.define_or_load_model(frame_seq_len, framelen, num_frame_seqs)


generator = Generator(utils, all_frames, seed_seq_len, utils.one_off_generate_len, learn_next_step)
generator.framelen = framelen

# generator seed can start at various positions in the frame set
# command line parameters can force this in the following call
utils.setup_seed_start(generator)

# for generating a model, no training iterations are required
# just generate the data from the model and exit
if utils.generate_mode():
  utils.log("Generating Samples")
  generator.generate(0)
  exit()



frame_rotate = 0

# train the model
# output generated frames after nth iteration
for iteration in range(start_iteration, num_iterations + 1):
  print('-' * 50)



  utils.iteration = iteration
  utils.log('Training Iteration', iteration)

  model_def.before_iteration(iteration)

  limit_frames = int(config.limit_frames)
  if limit_frames and limit_frames > 0:

    utils.log("frame rotate:", frame_rotate)
    utils.log("from frame:", frame_rotate*limit_frames)
    utils.log("to frame:", (frame_rotate+1)*limit_frames)

    Xl = X[frame_rotate*limit_frames : (frame_rotate+1)*limit_frames]
    Xl2 = X2[frame_rotate*limit_frames : (frame_rotate+1)*limit_frames]
    yl = y[frame_rotate*limit_frames : (frame_rotate+1)*limit_frames]
    yl2 = y2[frame_rotate*limit_frames : (frame_rotate+1)*limit_frames]
    utils.log("starting model fit with frames:", len(Xl))
  else:
    Xl = X
    Xl2 = X2
    yl = y
    yl2 = y2
    utils.log('using full set of frames')

  # if overlap_sequence == 0:
  #   inX = Xl
  # else:
  inX = [Xl, Xl2]

  outy = [yl, yl2]

  model_def.fit(inX, outy, batch_size=fit_batch_size, epochs=1, shuffle=config.shuffle,
   callbacks=[utils.csv_logger]
  )


  if save_model(iteration):
    print("saving .h5 model file")
    utils.save_h5_model(iteration)
    print("saving .h5 weights file")
    utils.save_weights(iteration)
    utils.write_iteration_count(iteration)
  else:
    print("not saving models this iteration")

  if gen_sequence(iteration):
    # every nth iteration generate sample data as a Codec 2 file

    utils.log("Generating samples")
    generator.generate(iteration)
  else:
    print("not generating samples this iteration")

  if limit_frames and limit_frames > 0:
    if (frame_rotate+1)*limit_frames > num_frame_seqs:
      frame_rotate=0
    else:
      utils.log("Rotate input to next frame set")
      frame_rotate+=1

  if config.stateful:
    utils.log("Reset states")
    model_def.model.reset_states()

  print()
