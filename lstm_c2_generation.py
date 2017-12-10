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
current_frame_seqs = []
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

# Check if we are in 'generate' mode.
if utils.generate_mode():
  # In generate mode
  num_frame_seqs = seed_seq_len
else:
  # In training mode

  # Pull the frames into frame sequences (frame_seqs), each of frame_seq_len frames
  # Each frame sequence is a sub-batch of timesteps, handed to the model in one chunk
  for i in range(0, num_frames - 2*frame_seq_len, seq_step):
    # The next frame starts frame_seq_len from the current index
    i_next = i + frame_seq_len
    # Store a set of frame sequences
    frame_seqs.append(all_frames[i : i_next])
    if learn_next_step:
        # Pull a single frame following each frame sequence into a corresponding array of next_frames
        # When just learning based on the next step after a frame sequence, the next frame is the one following
        # the last frame in the sequence
        next_frames.append(all_frames[i_next])
        # The current frame is therefore the last frame in the frame sequence
        current_frames.append(all_frames[i_next - 1 ])
    else:
        # If learning on a whole frame sequence, then start the sequence at the start of the next sequence
        # and make it span the same length
        next_frame_seqs.append(all_frames[i_next : (i_next + frame_seq_len)])
        # The current frame sequence is segmented in the same way as the input frame sequence in frame_seqs
        current_frame_seqs.append(all_frames[i : i_next])

  # Stateful operation requires the total set of timesteps to be a multiple of the batch size
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

  # Provide a second input set, containing sub-batch frame sequences that are shorter, to enable
  # 2D Convolutional networks to be trained without fake padding
  # These are effectively a window into the main frame sequence with a frame removed from each end
  # representing where the convolution is not able to reach
  X2 = np.zeros((num_frame_seqs, (frame_seq_len - overlap_sequence*2), framelen), dtype=np.float32)


  if learn_next_step:
      y = np.zeros((num_frame_seqs, framelen), dtype=np.float32)
      y2 = np.zeros((num_frame_seqs, framelen), dtype=np.float32)
  else:
      # If we are learning with a shortened X2 input sequence, the outputs we want to use for loss calculation will the
      # the same length as this shorter input
      y = np.zeros((num_frame_seqs, frame_seq_len - overlap_sequence*2, framelen), dtype=np.float32)
      y2 = np.zeros((num_frame_seqs, frame_seq_len - overlap_sequence*2, framelen), dtype=np.float32)



  for i, frame_seq in enumerate(frame_seqs):
      if learn_next_step:
          # expected output is always the next frame for corresponding frame_seq
          y[i] = next_frames[i]
          # The decoder output is always the current frame
          y2[i] = current_frames[i]
      else:
          if overlap_sequence != 0:
              y[i] = next_frame_seqs[i][overlap_sequence : frame_seq_len - overlap_sequence]
              y2[i] = current_frame_seqs[i][overlap_sequence : frame_seq_len - overlap_sequence]
          else:
              y[i] = next_frame_seqs[i]
              y2[i] = current_frame_seqs[i]

      # main input is simply each frame_seq
      X[i] = frame_seq
      # secondary shorter input takes the Conv2D unreachable frames off the start and finish, if we are using this
      if overlap_sequence != 0:
          X2[i] = frame_seq[overlap_sequence : frame_seq_len - overlap_sequence]
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


  inX = [Xl, Xl2]

  # Generate a mid layer encoded 'next step' output
  generator.input_frame_sequences = (Xl + Xl[0])[1:]
  out_mid = generator.generate_full_output(2)

  outy = [yl, yl2, out_mid]

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
    utils.write_gen_count(iteration)
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
