from __future__ import print_function

from model_utils import ModelUtils

# check command line args before loading everything, to save time

utils = ModelUtils()


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import Nadam # SGD #Adam #RMSprop
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import random
import time
import sys
import os

# number of training iterations
num_iterations = 600

fit_batch_size = 128
print("fit_batch_size: ", fit_batch_size)
# length of frame sequence to generate
genlen=400
print("genlen: ", genlen)
# generate sample data every nth iteration
gen_every_nth = 10

# number of bytes (unsigned 8 bit) in a Codec 2 frame
# note: one frame encodes 40ms of raw PCM audio
framelen=16

# length of frame sequence for learning
frame_seq_len = 40 # 200 # 5 seconds of audio
print("frame_seq_len: ", frame_seq_len)
step = 3

#frame_property_bits 
#1,1,1,1, # voiced flags for 4 PCM frames
#7, #Wo
#5, #E
#16,16,16,16,16,16,16,8,8,4 #LSP



frame_prop_loss_scale = np.array([
 64,64,64,64,
 1,
 4,
 8,8,8,8,8,8,8,16,16,32
])
print("frame_prop_loss_scale: ", str(frame_prop_loss_scale))

print("loading test data from: ", utils.testdata_filename)
testdata = np.fromfile(utils.testdata_filename, dtype=np.uint8)

len_testdata = len(testdata)

num_frames = int(len_testdata / framelen)

print('corpus length:', len_testdata)



# cut the testdata into sequences of frame_seq_len characters
print("scanning testdata into frames and frame sequences")

frame_seqs = []
next_frames = []
all_frames = []

# step through the testdata, pulling those bytes into an array of all the the frames, all_frames
for j in range(0, num_frames):
    i = j * framelen
    all_frames.append(testdata[i: i + framelen])

print('number of frames:', len(all_frames))

# pull the frames into frame sequences (frame_seqs) of frame_seq_len frames
# pull a single frame following each frame sequence into an array of next_frames
for i in range(0, num_frames - frame_seq_len, step):
    frame_seqs.append(all_frames[i: i + frame_seq_len])
    next_frames.append(all_frames[i + frame_seq_len])

print('number of frame sequences:', len(frame_seqs))


print('initialising input and expected output arrays')
X = np.zeros((len(frame_seqs), frame_seq_len, framelen), dtype=np.uint8)
y = np.zeros((len(frame_seqs), framelen), dtype=np.uint8)

for i, frame_seq in enumerate(frame_seqs):
    # expected output is always the next frame for corresponding frame_seq
    y[i] = next_frames[i]

    # input is just each frame_seq 
    X[i] = frame_seq
    #for t, frame in enumerate(frame_seq):
        #X[i,t] = frame


# Define a custom loss calculation, allowing the individual properties
# of the Codec 2 frame to be represented, based on their relative size
def codec2_param_error(y_true, y_pred):
  
  np.multiply(y_pred, frame_prop_loss_scale)
  np.multiply(y_true, frame_prop_loss_scale)
  
  # perform a basic mean absolute error calculation
  return K.mean(K.abs(y_pred - y_true), axis=-1)

# Define a new model
# only used if a model is not loaded from a file
def define_model():
    model =  Sequential()
    model.add(LSTM(
        20 
        ,input_shape=(frame_seq_len, framelen) 
#        ,return_sequences=True
      )
    ) 
#    model.add(LSTM(
#        40
#        , return_sequences=True
#        )
#      )
#    model.add(LSTM(40))

#    model.add(Dense(framelen))
    model.add(Dense(framelen))

    #model.add(Dropout(0.02))
    #model.add(Activation('relu'))
    
    optimizer = Nadam() #SGD() #Adam() #RMSprop(lr=0.01)
    loss = codec2_param_error #'mean_absolute_error'
    
    model.compile(loss=loss, optimizer=optimizer)
    return model

# process the sample prediction, ensuring it can be saved directly
# into a Codec 2 "charbits" file
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.round(preds)

    # it is necessary to cast to int before attempting to write to a file
    # to ensure that a real byte value is stored, not a byte 
    # representation of a floating point number
    intpreds = []
    for p in preds:
      intpreds.append(int(p))
    return np.array([intpreds], dtype=np.uint8) 

def gen_sequence(iteration):
  return (iteration % gen_every_nth == 0)

####  Setup the model
model = None

if len(utils.model_filename) > 0:

  utils.load_model()    
else:
  print("creating new model")
  model = define_model()
  utils.save_json_model(model)

# train the model
# output generated frames after nth iteration
for iteration in range(1, num_iterations + 1):
  print('-' * 50)
  print('Iteration', iteration)

  model.fit(X, y, batch_size=fit_batch_size, nb_epoch=1)

  if gen_sequence(iteration):
    ofn = utils.open_output_file(iteration)
    print("saving generated sample output to: ", ofn)
  else:
    print("not generating samples this iteration")  

  # every nth iteration generate sample data as a Codec 2 file
  if gen_sequence(iteration):
    print("generating sample data")
    start_index = random.randint(0, num_frames - frame_seq_len - 1)
    start_time = 1.0 * start_index / 40
    
    print("seed sequence for generation starts at frame index: ", start_index, " (approx. ", int(start_time / 60), ":", int(start_time % 60), ")" )

    # pick the seed frame sequence starting at the random start index, with frame_seq_len frames
    seed_frame_seq = all_frames[start_index: start_index + frame_seq_len]
    
    # the output file should start with a copy of the seed frame sequence
    for frame in seed_frame_seq:
      utils.output_file.write(frame)
      
    generated = []
    print('----- Generating with seed (just showing first): ', str(seed_frame_seq[0]) )
    
    for i in range(genlen):
      # setup seed input
      x = np.zeros((1, frame_seq_len, framelen))
      for t, frame in enumerate(seed_frame_seq):
        x[0, t] = frame

      # run the prediction for the next frame
      predicted_frame_props = model.predict(x, verbose=0)[0]
      # generate a Codec 2 frame from the predicted frame property values
      # we use the clumsy name predicted_frame_props to highlight that the frame properties are still
      # continuous (float) estimated values, rather than discrete Codec 2 values
      next_frame = sample(predicted_frame_props)
        
      # append the result to the generated set
      generated.append(next_frame)
      
      # update the seed frame sequence to remove the oldest frame and add the new predicted frame
      seed_frame_seq = seed_frame_seq[1:]
      seed_frame_seq.append(next_frame)

    # write the seed + generated data to the output file
    print ("writing output file to disk")
    for frame in generated:
      utils.output_file.write(frame)
        
    utils.output_file.close()
    
    if gen_sequence(iteration):
      print("saving .h5 model file")
      utils.save_h5_model(model, iteration)
    
  print()


