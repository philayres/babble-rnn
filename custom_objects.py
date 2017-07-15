from keras import backend as K
import numpy as np



frame_prop_loss_scale = [
   1,
   2**7,
   2**5,
   32,32,32,32,32,32,32,32,32,32
]


class CustomObjects:


  def __init__(self):
    print("frame_prop_loss_scale: ", str(frame_prop_loss_scale))

  # Define a custom loss calculation, allowing the individual properties
  # of the Codec 2 frame to be represented, based on their relative size

  @staticmethod
  def codec2_param_error(y_true, y_pred):
    
    y_pred = y_pred * frame_prop_loss_scale
    y_true = y_true * frame_prop_loss_scale
    diff_pred = y_pred - y_true
    # perform a basic mean absolute error calculation
    return K.mean(K.abs(diff_pred), axis=-1)
