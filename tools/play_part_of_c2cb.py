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

import numpy as np
import sys
from subprocess import check_output, call

infn = sys.argv[1]
outfn = sys.argv[2]
start = sys.argv[3]
length = sys.argv[4]
if not infn or not outfn:
  exit()

in_data = np.fromfile(infn, dtype=np.uint8)
# in_data = in_data.astype(int)
shaped_data = np.reshape(in_data, (-1,13))


startv = int(start)
lengthv = int(length)
endv = startv + lengthv
segment = shaped_data[startv : endv]

# play with blanking columns just for fun (and learning)

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

segment = np.reshape(segment, (1,-1))
np.savetxt(outfn, segment.astype(np.uint8), delimiter="", fmt="%c")

call(["bash", "./c2towav.sh", outfn])
call(["cvlc", outfn + "-3200.wav"])
