#!/usr/bin/python
#
# Play a segment of a c2cb file. Uses vlc if available.
#
# Usage:
# tools/play_part_of_c2cb.py infilename outfilename startframe endframe
#
# tools/play_part_of_c2cb.py test/copy-criticalpracticalreason.c2-3200 /tmp/cc2 5000 1000
#
#

# Parameters for c2cb files are one byte (although most only use part of the range)
# 0: voicing
# 1: Wo fundamental frequency
# 2: Energy
# 3-12: remaining frequency components

# The following are the maximum values for each parameter:

frame_prop_scale = [
   1,
   127,
   31 ,
   31,31,31,31,31,31,31,31,31,31
]

import numpy as np
import sys
from subprocess import check_output, call

infn = sys.argv[1]
outfn = sys.argv[2]
start = sys.argv[3]
length = sys.argv[4]
if len(sys.argv) > 5:
  filter = sys.argv[5]
else:
  filter = None
if not infn or not outfn:
  exit()

in_data = np.fromfile(infn, dtype=np.uint8)
# in_data = in_data.astype(int)
shaped_data = np.reshape(in_data, (-1,13))


startv = int(start)
lengthv = int(length)
endv = startv + lengthv
segment = shaped_data[startv : endv]

# play with filters just for fun (and learning)
if filter:
  print("Using filter")
  # daleks, anyone?
  #segment[:,1] = np.zeros(lengthv)

  # lose some frequencies
  # (watch your ears - some are nasty!)
  # segment[:,12:13] = np.zeros((lengthv,1))


  # Fundamental frequency, voiced with constant amplitude
  # segment[:,0] = np.ones(lengthv)
  # segment[:,2] = np.ones(lengthv)
  # segment[:,3:] = np.zeros((lengthv, 10))

  # High pitch
  # segment[:,1] = segment[:,1] +25

  # Compressed fundamental frequency
  # m = np.mean(segment[:,1])
  # segment[:,1] = m + (segment[:,1] - m) / 3

  # Clip diff from mean
  # m = np.mean(segment[:,1])
  # deltas = segment[:,1] - (m)
  # segment[:,1] = np.clip(deltas, -31, 31) + m
  # print("Clipped total:", np.sum(np.abs(deltas - np.clip(deltas, -31, 31))))

  # Block average of a parameter
  params = [0,3,4,5,6,7,8,9,10,11,12]
  block_size = 2
  for param in params:
    ff = segment[:,param]
    blocked = np.reshape(ff, (-1, block_size))
    mean_blocked = np.mean(blocked, axis=-1)

    res = np.repeat(mean_blocked, block_size)

    segment[:,param] = res


# Clip the output to ensure we don't exceed the parameter maximums
for i in range(13):
  segment[:,i] = np.clip(segment[:,i], 0, frame_prop_scale[i])

segment = np.reshape(segment, (1,-1))
np.savetxt(outfn, segment.astype(np.uint8), delimiter="", fmt="%c")

call(["bash", "./c2towav.sh", outfn])
call(["cvlc", outfn + "-3200.wav"])
