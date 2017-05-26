The network _tale-prog-large6_ is a three layer LSTM with a final dense layer of ReLU cells.

The network was set to learn from the model: 
  tale-prog-large3/model-460.h5

All three LSTM layers were made trainable. The process was only run for 10 iterations in order to get a fast view of whether loss improvements could be rapidly achieved.

Optimizer: Nadam (Nesterov Adam), with default parameters (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

Batches per iteration (single epoch per iteration): 72880 batches of 200 Codec2 frames (equivalent to 5 seconds of audio).


