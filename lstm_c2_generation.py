'''Example 


'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import os
import time

framelen=16
epoch_time = int(time.time())

fname="test/LDC97S44-8k.c2cb"
odir="out/LDC97S44/"+str(epoch_time)+"/"
os.makedirs(odir)
ofname=odir+"out-c2cb-"
modelf=odir+"model-"
#text = open(fname).read()
text = np.fromfile(fname, dtype=np.uint8)
print('corpus length:', len(text))


len_text = len(text)
#len_text = 16000

'''
bytes=[]
for i,t in enumerate(text):
    bytes.append(ord(t))
text = bytes
'''

num_frames = int(len_text / framelen)
#chars = sorted(list(set(text)))
#print('total chars:', len(chars))
#char_indices = dict((c, i) for i, c in enumerate(chars))
#indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 100
step = 3
sentences = []
next_chars = []
print("scanning byte sequence")

frames = []

for j in range(0, num_frames):
    i = j*framelen
    frames.append(text[i: i+framelen])
print('nb frames:', len(frames), len_text/framelen)
print("scanning frames of length ", len(frames[0]))
for i in range(0, num_frames-maxlen, step):
    # print("getting set from:", i)
    sentences.append(frames[i: i + maxlen])
    next_chars.append(frames[i + maxlen])
print("got sets")
print('nb sequences in bytes:', len(sentences))


print('Vectorization...')
X = np.zeros((len(sentences), maxlen, framelen), dtype=np.uint8)
y = np.zeros((len(sentences), framelen), dtype=np.uint8)
for i, sentence in enumerate(sentences):
    for t, frame in enumerate(sentence):
        #X[i, t, char_indices[char]] = 1
        #for p, prop in enumerate(frame):
         #   X[i, t, p] = (prop)
        X[i,t] = frame
    #y[i, char_indices[next_chars[i]]] = 1
    #for n in range(0, framelen):
    #  y[i, n] = (next_chars[i][n])
    y[i] = next_chars[i]


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, framelen)))
model.add(Dense(framelen))
#model.add(LSTM(128))
#model.add(Dense(framelen))
model.add(Dense(framelen))

model.add(Dropout(0.2))
#model.add(Dense(framelen))
#model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='mean_absolute_error', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.round(preds)
   # preds = np.log(preds) / temperature
  #  exp_preds = np.exp(preds)
#    preds = exp_preds / np.sum(exp_preds)
#    probas = np.random.multinomial(1, preds, 1)
    return preds # np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 600):
    ofile= open(ofname+str(iteration), "w")
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, num_frames - maxlen - 1)
    print("start index and maxlen from frames: ", start_index, maxlen, num_frames)

    generated = []
    sentence = frames[start_index: start_index + maxlen]
    generated += ( sentence)
    print('----- Generating with seed: ', str(sentence[0]) )
   # sys.stdout.write(str(generated[0:1]))
  #  sys.stdout.flush()
    
    for i in range(400):
        x = np.zeros((1, maxlen, framelen))
     #   print("* from range: ", i)
    #    print("sentence length", len(sentence))
        for t, frame in enumerate(sentence):
            #for p in range(0, framelen):
            #    f = frame[p]
               # x[0, t, p] = float(ord(f))
            x[0, t] = frame

        preds = model.predict(x, verbose=0)[0]
        next_frame = sample(preds)
        
 #           next_char = indices_char[next_index]
     #   intpreds=[]
  #      for p in np.nditer(preds):
   #         intpreds.append(float(round(p)))
   #     next_frame = intpreds
    #    sys.stdout.write(str(next_frame))
    #    sys.stdout.flush()
        generated.append( next_frame)
        sentence = sentence[1:]
        sentence.append( next_frame)
       # print("new sentence length", len(sentence))
   
#    print("last result: ", str(generated[-1]) )
    for i, frame in enumerate(generated):
      for j,c in enumerate(frame):
        ic = int(c)
        if ic>255:
           ic = 255
        if ic<0:
           ic = 0
        try:
          ofile.write(chr(ic))
          break
        except ValueError:
          ofile.write(chr(0))
          print("failed to store value: ", ic)
#        print(str(frame))

      #for j,c in enumerate(frame):
 #       ofile.write((frame))
        
    ofile.close()
    model.save(modelf+str(iteration))
    print()
