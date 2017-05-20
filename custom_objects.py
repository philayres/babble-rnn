from keras import backend as K
import numpy as np

#frame_property_bits 
#1,1,1,1, # voiced flags for 4 PCM frames
#7, #Wo
#5, #E
#16,16,16,16,16,16,16,8,8,4 #LSP

frame_prop_loss_scale = [
 1,1,1,1,
 2**7,
 2**5,
 16,16,16,16,16,16,16,8,8,4
]

'''
([
   16,16,16,16,
   1,
   4,
   8,8,8,8,8,8,8,16,16,32
  ])
'''

class CustomObjects:


  def __init__(self, utils):
    utils.log("frame_prop_loss_scale: ", str(frame_prop_loss_scale))

  # Define a custom loss calculation, allowing the individual properties
  # of the Codec 2 frame to be represented, based on their relative size

  @staticmethod
  def codec2_param_error(y_true, y_pred):
    
    y_pred = y_pred * frame_prop_loss_scale
    y_true = y_true * frame_prop_loss_scale
    diff_pred = y_pred - y_true
    # perform a basic mean absolute error calculation
    return K.mean(K.abs(diff_pred), axis=-1)
