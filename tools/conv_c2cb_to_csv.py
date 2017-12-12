#!/usr/bin/python
#
# Simply convert a 3200 bit rate Codec2 (c2cb format) file to CSV
#
# Usage:
# ./conv_c2cb_to_csv.py infilename outfilenam


import numpy as np
import sys


infn = sys.argv[1]
outfn = sys.argv[2]

if not infn or not outfn:
  exit()

in_data = np.fromfile(infn, dtype=np.uint8)
in_data = in_data.astype(int)
shaped_data = np.reshape(in_data, (-1,13))

np.savetxt(outfn, shaped_data, delimiter=",", fmt="%u")
