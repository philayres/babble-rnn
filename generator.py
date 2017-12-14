import random
import numpy as np
from subprocess import call

class Generator:

  generate_with_single_timestep = True
  config = None
  utils = None
  all_frames = None
  seed_seq_len = None
  generate_len = None
  framelen = None
  num_frames = None
  seed_start_index = 60
  frame_len_ms = None
  input_frame_sequences = None

  def __init__(self, utils, all_frames, seed_seq_len=None, generate_len=None, generate_with_single_timestep=None):
    self.utils = utils
    self.config = utils.config
    self.all_frames = all_frames
    if seed_seq_len is not None:
      self.seed_seq_len = seed_seq_len
    else:
      self.seed_seq_len = self.config.seed_seq_len
    if generate_len is not None:
      self.generate_len = generate_len
    else:
      self.generate_len = self.config.generate_len
    utils.log("generate_len:", generate_len)
    self.num_frames = len(all_frames)
    if generate_with_single_timestep is not None:
      self.generate_with_single_timestep = generate_with_single_timestep
    else:
      self.generate_with_single_timestep = self.config.learn_next_step
    self.frame_len_ms = self.config.frame_len_ms
    self.seed_start_index = self.config.seed_start_index


  def set_random_seed_start_index(self):
    self.seed_start_index = random.randint(0,
      self.num_frames - self.seed_seq_len - 1)
    self.fix_seed_start_index()

  def set_time_seed_start_index(self, seconds):

    self.seed_start_index = int(float(seconds) / self.frame_len_ms)
    self.fix_seed_start_index()

  def set_frame_seed_start_index(self, index):
    self.seed_start_index = int(index)
    self.fix_seed_start_index()

  def fix_seed_start_index(self):
    if self.seed_start_index < 0: self.seed_start_index = 0
    latest = self.num_frames - self.seed_seq_len - 1
    if self.seed_start_index > latest: self.seed_start_index = latest

  # process the sample prediction, ensuring it can be saved directly
  # into a Codec 2 "charbits" file
  def sample(self, preds, no_scale=False):
    preds = np.asarray(preds).astype('float32')
    if not no_scale:
      preds = [min(1, a) for a in preds]
      preds = np.multiply(preds, self.config.frame_prop_orig_scale)

    preds = np.round(preds)

    # it is necessary to cast to int before attempting to write to a file
    # to ensure that a real byte value is stored, not a byte
    # representation of a floating point number
    intpreds = []
    for p in preds:
      # rectify, just in case the final dense layer produces negatives
      q = max(int(p), 0)
      intpreds.append(q)
    return np.array([intpreds], dtype=np.uint8)

  def generate_full_output(self, output_index = 2):
    utils = self.utils
    model_def = utils.model_def

    print("Generating full output for output index:", output_index)
    self.generated_output = model_def.model.predict(self.input_frame_sequences, batch_size=len(self.input_frame_sequences))[output_index]

    return self.generated_output


  def generate(self, iteration):
    utils = self.utils
    all_frames = self.all_frames
    seed_seq_len = self.seed_seq_len
    generate_len = self.generate_len
    framelen = self.config.framelen
    num_frames = self.num_frames
    overlap_sequence = self.config.overlap_sequence
    generate_num_outputs = self.config.generate_num_outputs
    frame_seq_len = self.config.frame_seq_len
    use_short_input = False

    model_def = utils.model_def

    for outi in range(generate_num_outputs):

        fn_postfix = "output_" + str(outi) + "_" + str(iteration)

        ofn = utils.open_output_file(fn_postfix)
        utils.log("saving generated sample output to: ", ofn)

        utils.log("generating sample data of length: ", generate_len)
        start_index = self.seed_start_index
        start_time = 1.0 * start_index / self.frame_len_ms

        utils.log("seed sequence for generation starts at frame index: ", start_index, " (approx. ", int(start_time / 60), ":", int(start_time % 60), ")" )
        utils.log("seed sequence length:",  seed_seq_len)
        # pick the seed frame sequence starting at the random start index, with seed_seq_len frames
        seed_frame_seq = all_frames[start_index: start_index + seed_seq_len]

        # the output file should start with a copy of the seed frame sequence
        for frame in seed_frame_seq:
          utils.output_file.write(self.sample(frame).tostring())

        generated = []
        print('----- Generating with seed (just showing first): ', str(seed_frame_seq[0]) )


        if self.generate_with_single_timestep:
          loop_len = generate_len
          utils.log('generate with single timesteps:', loop_len)
        else:
          loop_len = int(generate_len /  frame_seq_len)
          utils.log('generate with frame sequences:', loop_len)

        for i in range(loop_len):
          if utils.generate_mode():
            print("Generating", i, "of", generate_len)
          # setup seed input
          x = np.zeros((1, seed_seq_len, framelen), dtype=np.float32)
        #   if overlap_sequence != 0:
          x2 = np.zeros((1, seed_seq_len - overlap_sequence*2, framelen), dtype=np.float32)
          for t, frame in enumerate(seed_frame_seq):
            x[0, t] = frame

            # Handle the shortened sequence
            if overlap_sequence == 0:
                x2[0, t] = frame
            else:
                # Ignore the first few timesteps
                if t >= overlap_sequence and t < frame_seq_len - overlap_sequence:
                    # Add the frames, starting at the beginning of the shortened array
                    x2[0, t-overlap_sequence] = frame

          if use_short_input:
            inx = [x, x2]
          else:
            inx = x

          if utils.generate_mode() : utils.log("predicting",i)
          # run the prediction for the next frame, getting the result
          # from the specified output, outi
          all_predicted_frame_props = model_def.model.predict_on_batch(inx)

          # Write out a specific output to file, for debugging purposes
          # wf = open('workfile.txt', 'w')
          # for r in all_predicted_frame_props[2]:
          #   for s in r:
          #     for sn in s:
          #       wf.write(str(sn)+' ')
          #     wf.write("\n")

          predicted_frame_props = all_predicted_frame_props[outi]

          if loop_len > 0:
            # predicted_frame_props = model_def.model.predict(x,
            # batch_size=self.generate_len, verbose=0)[0]
            # generate a Codec 2 frame from the predicted frame property values
            # we use the clumsy name predicted_frame_props to highlight that the frame properties are still
            # continuous (float) estimated values, rather than discrete Codec 2 values

            next_frame = predicted_frame_props

            # append the result to the generated set
            generated.append(next_frame)

            if self.generate_with_single_timestep:
              # update the seed frame sequence to remove the oldest frame and add the new predicted frame
              seed_frame_seq = seed_frame_seq[1:]
              seed_frame_seq.append(next_frame)
            else:
              utils.log("using generated frames as seed_seq:", next_frame.shape)
              if len(next_frame.shape) == 3 and next_frame.shape[0] == 1:
                  seed_frame_seq = next_frame[0]
              else:
                  seed_frame_seq = next_frame

          else:
            # Final loop. No need to setup seed again
            for i in predicted_frame_props:
              # take all the results and append them to the generated array
              generated.append(i)

        # We are done generating predictions
        # write the seed + generated data to the output file
        print("writing output file to disk")
        for frame in generated:
          # if we are passing multiple frames
          #(stateful or time distributed operation with learn_next_step = False)
          if len(frame.shape) > 1:
            utils.log("Generated multiple frames in one action:", frame.shape, frame.shape[0])
            if len(frame.shape) == 3 and frame.shape[0] == 1:
                for f in frame[0]:
                  s = self.sample(f).tostring()
                  utils.output_file.write(s)
            else:
                for f in frame:
                  s = self.sample(f).tostring()
                  utils.output_file.write(s)
          else:
            # just one frame at a time
            s = self.sample(frame).tostring()
            utils.output_file.write(s)

        utils.output_file.close()
        utils.log("wrote frames: ", len(generated))

        if utils.generate_mode():
          utils.log("converting:", utils.output_fn)
          call(["bash", "./c2towav.sh", utils.output_fn])
