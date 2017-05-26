The network _tale-prog-large3_ is a three layer LSTM with a final dense layer of ReLU cells.

The network was set to learn as follows:

Iterations   1 -  59: LSTM 1 & Dense trainable

Iterations  60 - 239: LSTM 1, LSTM 2 & Dense trainable

Iterations 240 - 600: LSTM 3 & Dense trainable

Optimizer: Nadam (Nesterov Adam), with default parameters (lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

Batches per iteration (single epoch per iteration): 72880 batches of 200 Codec2 frames (equivalent to 5 seconds of audio).


