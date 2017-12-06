from keras import backend as K

class CustomObjects:

  frame_prop_loss_scale = None


  # Define a custom loss calculation, allowing the individual properties
  # of the Codec 2 frame to be represented, based on their relative size

  @staticmethod
  def codec2_param_error_to_end(y_true, y_pred):

    y_pred = y_pred * CustomObjects.frame_prop_loss_scale
    y_true = y_true * CustomObjects.frame_prop_loss_scale
    diff_pred = K.square(y_pred) - K.square(y_true)
    # perform a basic mean absolute error calculation
    return K.mean(K.abs(diff_pred), axis=-1)


  @staticmethod
  def codec2_param_error(y_true, y_pred):

    y_pred = y_pred * CustomObjects.frame_prop_loss_scale
    y_true = y_true * CustomObjects.frame_prop_loss_scale
    diff_pred = y_pred - y_true
    # perform a basic mean absolute error calculation
    return K.mean(K.abs(diff_pred), axis=-1)

  @staticmethod
  def codec2_param_error_td(y_true, y_pred):
    # time distributed version that attempts to calculate loss across the fully time distributed input / output data sequence
    y_pred = y_pred * CustomObjects.frame_prop_loss_scale
    y_true = y_true * CustomObjects.frame_prop_loss_scale
    diff_pred = y_pred - y_true
    # perform a basic mean absolute error calculation
    return K.mean(K.abs(diff_pred))

  @staticmethod
  def codec2_param_mean_square_error(y_true, y_pred):

    y_pred = y_pred * CustomObjects.frame_prop_loss_scale
    y_true = y_true * CustomObjects.frame_prop_loss_scale
    diff_pred = y_pred - y_true
    # perform a basic mean absolute error calculation
    return K.mean(K.square(diff_pred), axis=-1)
